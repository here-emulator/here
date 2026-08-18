use std::{
    hint::cold_path,
    ops::{Deref, DerefMut},
};

use crate::{
    board::virt::RiscvIRQHandler,
    config::arch_config::WordType,
    cpu::RegFile,
    fpu::soft_float::SoftFPU,
    isa::{
        InstrLen,
        cache::{Cache, CachePolicy, DirectCache},
        riscv::{
            RawInstr,
            csr_reg::{CsrRegFile, NamedCsrReg, PrivilegeLevel, csr_macro::*},
            decoder::{DecodeInstr, Decoder},
            instruction::{RVInstrInfo, exec_mapping::get_exec_func, instr_table::RiscvInstr},
            mmu::VirtAddrManager,
            trap::{Exception, Interrupt, trap_controller::TrapController},
            vector::Vector,
        },
    },
    ram_config::DEFAULT_PC_VALUE,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchResult {
    pub cycles: u64,
    pub hook_stopped: bool,
}

pub trait ExecutorBackend {
    /// Execute exactly `cycles` cycles.
    fn step_batch(&mut self, cpu: &mut RVCPU, cycles: u64);
}

#[derive(Default)]
pub struct InterpreterBackend;

impl ExecutorBackend for InterpreterBackend {
    fn step_batch(&mut self, cpu: &mut RVCPU, cycles: u64) {
        cpu.step_batch(cycles);
    }
}

pub trait ExecutionHook {
    /// Per-cycle data carried from before execution to [`ExecutionHook::after_step`].
    type CycleContext;

    fn before_step(&mut self, cpu: &mut RVCPU) -> Self::CycleContext;
    fn on_interrupt_taken(&mut self, interrupted_pc: WordType) -> Self::CycleContext;

    /// Return `true` to stop the current batch after this cycle.
    fn after_step(&mut self, context: Self::CycleContext, cpu: &mut RVCPU) -> bool;
}

pub(crate) struct NoopExecutionHook;

impl ExecutionHook for NoopExecutionHook {
    type CycleContext = ();

    #[inline(always)]
    fn before_step(&mut self, _cpu: &mut RVCPU) {}

    #[inline(always)]
    fn on_interrupt_taken(&mut self, _interrupted_pc: WordType) {}

    #[inline(always)]
    fn after_step(&mut self, _context: (), _cpu: &mut RVCPU) -> bool {
        false
    }
}

#[repr(C)]
pub struct CpuContext {
    pub(crate) reg_file: RegFile,
    pub(crate) pc: WordType,
}

pub struct RVCPU {
    context: CpuContext,
    pub(super) memory: VirtAddrManager,
    pub(super) decoder: Decoder,
    pub(super) csr: CsrRegFile,
    pub(super) icache: Cache<DirectCache<DecodeInstr, 8192>>,
    icache_epoch: u64,
    pub(super) fpu: SoftFPU,
    pub(super) vector: Vector,

    /// The address of the memory-mapped `mtime` CSR.
    pub(crate) time_addr: Option<WordType>,

    /// The trap value pending to be written to `mtval`/`stval`.
    pub(super) pending_tval: Option<WordType>,
}

// A workaround so that legacy code can use `cpu.pc` to access `cpu.context.pc`.
impl Deref for RVCPU {
    type Target = CpuContext;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.context
    }
}

impl DerefMut for RVCPU {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.context
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExceptionInfo {
    cause: Exception,
    tval: WordType,
}

impl Drop for RVCPU {
    fn drop(&mut self) {
        log::info!("iCache hit rate {}", self.icache.hit_rate());
    }
}

impl RVCPU {
    pub(crate) fn from_vaddr_manager(v_memory: VirtAddrManager) -> Self {
        Self::from_decoder(Decoder::new(), v_memory)
    }

    pub(crate) fn from_decoder(decoder: Decoder, v_memory: VirtAddrManager) -> Self {
        let mut csr = CsrRegFile::new();

        let ext = decoder.extension_bits();
        csr.ctx.extension = ext;

        let mxl = if WordType::BITS == 32 {
            1
        } else {
            debug_assert!(WordType::BITS == 64);
            2
        };

        csr.get_by_type_existing::<Misa>()
            .set_extension_directly(ext);
        csr.get_by_type_existing::<Misa>().set_mxl_directly(mxl);
        csr.get_by_type_existing::<Mstatus>().set_sxl_directly(mxl);
        csr.get_by_type_existing::<Mstatus>().set_uxl_directly(mxl);

        debug_assert!(csr.get_by_type_existing::<Mstatus>().get_uxl() == mxl);
        debug_assert!(csr.get_by_type_existing::<Mstatus>().get_sxl() == mxl);
        debug_assert!(csr.get_by_type_existing::<Sstatus>().get_uxl() == mxl);

        csr.set_current_privileged(PrivilegeLevel::M);

        let fpu = SoftFPU::from(true);

        Self {
            context: CpuContext {
                reg_file: RegFile::new(),
                pc: DEFAULT_PC_VALUE,
            },
            memory: v_memory,
            decoder,
            csr: csr,
            vector: Vector::new(),
            icache: Cache::new(),
            icache_epoch: 0,
            fpu,
            time_addr: None,
            pending_tval: None,
        }
    }

    pub(in super::super) fn execute(
        &mut self,
        instr: RiscvInstr,
        info: RVInstrInfo,
    ) -> Result<(), Exception> {
        let rst = get_exec_func(instr)(info, self);
        self.reg_file[0] = 0;

        if let Err(ex) = rst {
            cold_path();

            if ex == Exception::IllegalInstruction {
                log::warn!(
                    "IllegalInstruction for instr: {:#?} at pc = {:#x}, info: {:?} ",
                    instr,
                    self.pc,
                    info,
                );
            }
        }

        rst
    }

    pub fn read_csr(&mut self, addr: WordType) -> Result<WordType, Exception> {
        if addr == 0xc01 {
            // time CSR
            if let Some(time_addr) = self.time_addr {
                if let Ok(time) = self.memory.read_by_paddr::<u64>(time_addr) {
                    return Ok(time as WordType);
                }
            }
        } else if let Some(data) = self.csr.read(addr) {
            // Normal CSR read
            return Ok(data);
        }

        Err(Exception::IllegalInstruction)
    }

    /// Write CSR and update context correctly.
    ///
    /// XXX: Use this function instead of `self.csr.write`, unless you are sure about what you are doing.
    ///
    /// You may need [`CsrRegFile::write_directly`] in some cases.
    pub fn write_csr(&mut self, addr: WordType, data: WordType) -> Result<(), Exception> {
        if !self.csr.write(addr, data) {
            log::warn!("Failed to write CSR {:#x} with data {:#x}", addr, data);
            return Err(Exception::IllegalInstruction);
        }

        // Changing satp.MODE takes effect immediately, without SFENCE.VMA.
        if addr == Satp::get_index() {
            let satp = self.csr.get_by_type_existing::<Satp>();
            self.memory.set_mode(satp.get_mode() as u8);
            self.memory.set_root_ppn(satp.get_ppn() as u64);
            self.memory.flush_tlb();
            self.flush_icache();
        }

        Ok(())
    }

    /// Execute one cycle. This may be slower than batching; prefer
    /// [`RVCPU::step_batch`] or [`RVCPU::step_batch_with_hook`] when possible.
    pub fn step(&mut self) {
        self.step_batch(1);
    }

    pub fn step_batch(&mut self, cycles: u64) {
        let mut hook = NoopExecutionHook;
        self.step_batch_with_hook(cycles, &mut hook);
    }

    #[inline]
    pub(crate) fn step_batch_no_interrupt(&mut self, cycles: u64) {
        let mut executed = 0;
        while executed < cycles {
            self.step_impl();
            self.increment_mcycle();
            executed += 1;

            debug_assert!(self.pending_tval.is_none());
        }
    }

    pub fn step_batch_with_hook<H: ExecutionHook>(
        &mut self,
        cycles: u64,
        hook: &mut H,
    ) -> BatchResult {
        if cycles == 0 {
            return BatchResult {
                cycles: 0,
                hook_stopped: false,
            };
        }

        let mut executed = 0;
        let interrupt_pc = self.pc;
        if TrapController::try_take_interrupt(self).is_some() {
            let context = hook.on_interrupt_taken(interrupt_pc);
            self.increment_mcycle();
            executed += 1;

            if hook.after_step(context, self) {
                return BatchResult {
                    cycles: executed,
                    hook_stopped: true,
                };
            }
        }

        while executed < cycles {
            let context = hook.before_step(self);
            self.step_impl();
            self.increment_mcycle();
            executed += 1;

            debug_assert!(self.pending_tval.is_none());

            if hook.after_step(context, self) {
                return BatchResult {
                    cycles: executed,
                    hook_stopped: true,
                };
            }
        }

        BatchResult {
            cycles: executed,
            hook_stopped: false,
        }
    }

    fn increment_mcycle(&mut self) {
        self.increment_mcycle_by(1);
    }

    fn increment_mcycle_by(&mut self, cycles: u64) {
        let mcycle = self.csr.get_by_type_existing::<Mcycle>();
        mcycle.set_mcycle_directly(mcycle.data().wrapping_add(cycles as WordType));
    }

    pub(crate) fn try_take_pending_interrupt(&mut self) -> bool {
        if TrapController::try_take_interrupt(self).is_none() {
            return false;
        }

        self.increment_mcycle();
        true
    }

    pub(crate) fn finish_jit_block(&mut self, next_pc: WordType, instr_count: u64) {
        self.pc = next_pc;
        self.reg_file[0] = 0;
        self.increment_mcycle_by(instr_count);
        self.csr
            .get_by_type_existing::<Minstret>()
            .wrapping_add(instr_count as WordType);
    }

    pub(crate) fn context_mut(&mut self) -> &mut CpuContext {
        &mut self.context
    }

    pub(crate) fn icache_epoch(&self) -> u64 {
        self.icache_epoch
    }

    fn ifetch(&mut self) -> Result<RawInstr, ExceptionInfo> {
        self.ifetch_at(self.pc)
    }

    fn ifetch_at(&mut self, addr: WordType) -> Result<RawInstr, ExceptionInfo> {
        let mut bytes: RawInstr =
            (self
                .memory
                .ifetch::<u16>(addr, &mut self.csr)
                .map_err(|err| ExceptionInfo {
                    cause: Exception::from_instr_fetch_err(err),
                    tval: addr,
                })? as u32)
                .into();

        if bytes.len() == 4 {
            // 32-bit instr.

            // "The C extension allows 16-bit instructions to be freely intermixed with 32-bit instructions,
            // with the latter now able to start on any 16-bit boundary."

            // but the next half may sit on the next page, causing a page fault.
            let next_half = match self.memory.ifetch::<u16>(addr + 2, &mut self.csr) {
                Ok(half) => half as u32,
                Err(err) => {
                    return Err(ExceptionInfo {
                        cause: Exception::from_instr_fetch_err(err),
                        tval: addr.wrapping_add(2),
                    });
                }
            };
            bytes.val |= next_half << 16;
        };

        Ok(bytes)
    }

    #[inline]
    pub(crate) fn decode_at(&mut self, addr: WordType) -> Option<DecodeInstr> {
        let raw = self.ifetch_at(addr).ok()?;
        self.decoder.decode(raw)
    }

    fn step_impl(&mut self) {
        let DecodeInstr {
            instr,
            info,
            len: _,
        } = if let Some(decode_instr) = self.icache.get(self.pc) {
            decode_instr
        } else {
            let raw_instr = match self.ifetch() {
                Ok(bytes) => bytes,
                Err(err) => {
                    TrapController::take_exception(self, err.cause, err.tval);
                    return;
                }
            };

            // ID
            let decoder_result = self.decoder.decode(raw_instr);
            let Some(decode_instr) = decoder_result else {
                cold_path();
                log::warn!(
                    "Illegal instruction: {:#x} at {:#x}",
                    raw_instr.val,
                    self.pc
                );
                TrapController::take_exception(
                    self,
                    Exception::IllegalInstruction,
                    raw_instr.val as WordType,
                );
                return;
            };

            self.icache.put(self.pc, decode_instr.clone());
            decode_instr
        };

        // EX && MEM && WB
        let excute_result = self.execute(instr, info);
        match excute_result {
            // XXX: OpenSBI have semihosting test, and we don't implement breakpoint exception handling yet,
            // so we can't throw and panic here.
            // Err(Exception::Breakpoint) => return excute_result,
            Err(Exception::IllegalInstruction) => {
                cold_path();

                // We cannot reuse the fetched raw instruction on the i-cache hit path,
                // because the raw instruction bytes are not stored in the i-cache.
                // This is acceptable because `illegal instruction` is a cold path.
                let raw_instr = self.ifetch().expect("ifetch should not fail here");
                TrapController::take_exception(
                    self,
                    Exception::IllegalInstruction,
                    raw_instr.val as WordType,
                );
            }
            Err(nr) => {
                let tval = self.pending_tval.take().unwrap_or(0);
                TrapController::take_exception(self, nr, tval);
            }
            Ok(()) => {}
        }
    }

    pub(crate) fn flush_icache(&mut self) {
        self.icache.clear();
        self.icache_epoch = self.icache_epoch.wrapping_add(1);
    }

    pub fn flush_tlb(&mut self) {
        self.memory.flush_tlb();
    }

    pub fn power_off(&mut self) {
        self.memory.sync();
    }
}

#[cfg(test)]
mod context_layout_test {
    use super::*;

    #[test]
    fn cpu_context_layout_is_stable_for_codegen() {
        assert_eq!(std::mem::offset_of!(CpuContext, reg_file), 0);
        assert_eq!(
            std::mem::offset_of!(CpuContext, pc),
            std::mem::size_of::<RegFile>()
        );
        assert_eq!(
            std::mem::size_of::<CpuContext>(),
            std::mem::size_of::<RegFile>() + std::mem::size_of::<WordType>()
        );
    }
}

impl RiscvIRQHandler for RVCPU {
    fn handle_irq(&mut self, interrupt: Interrupt, level: bool) {
        let mip = self.csr.get_by_type_existing::<Mip>();
        let level = level as WordType;

        match interrupt {
            Interrupt::MachineTimer => {
                mip.set_mtip(level);
            }
            Interrupt::MachineExternal => {
                mip.set_meip(level);
            }
            Interrupt::SupervisorExternal => {
                mip.set_seip(level);
            }

            Interrupt::MachineSoft => {
                mip.set_msip(level);
            }

            _ => {
                todo!("IRQ handling not implemented yet.")
            }
        }
    }
}

#[cfg(test)]
#[path = "cpu_test.rs"]
mod test;
