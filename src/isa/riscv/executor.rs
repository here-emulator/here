use std::hint::cold_path;

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
pub struct RVCPU {
    pub(crate) reg_file: RegFile,
    pub(crate) pc: WordType,
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
    ///
    /// [`TrapController`] won't read this field, take that and pass by argument.
    pub(super) pending_tval: Option<WordType>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExceptionInfo {
    cause: Exception,
    tval: WordType,
}

impl Drop for RVCPU {
    fn drop(&mut self) {
        log::info!("iCache hit rate: {}", self.icache.hit_rate());
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
            reg_file: RegFile::new(),
            pc: DEFAULT_PC_VALUE,
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

    pub(crate) fn finish_jit_block(
        &mut self,
        next_pc: WordType,
        instr_count: u64,
        exception: Option<(Exception, WordType)>,
    ) {
        self.pc = next_pc;
        self.increment_mcycle_by(instr_count + u64::from(exception.is_some()));
        self.csr
            .get_by_type_existing::<Minstret>()
            .wrapping_add(instr_count as WordType);

        if let Some((exception, tval)) = exception {
            // Memory helpers still call the interpreter-facing read/write APIs, which may leave
            // a compatibility tval behind. JIT exception data comes exclusively from JitContext.
            self.pending_tval = None;
            TrapController::take_exception(self, exception, tval);
        } else {
            debug_assert!(self.pending_tval.is_none());
        }
    }

    pub(crate) fn icache_epoch(&self) -> u64 {
        self.icache_epoch
    }

    pub(crate) fn instruction_alignment(&mut self) -> WordType {
        if self.csr.get_by_type_existing::<Misa>().c_enabled() {
            2
        } else {
            4
        }
    }

    #[cfg(test)]
    pub(crate) fn set_privilege_for_test(&mut self, privilege: PrivilegeLevel) {
        self.csr.set_current_privileged(privilege);
    }

    fn ifetch(&mut self) -> Result<RawInstr, ExceptionInfo> {
        self.ifetch_at(self.pc)
    }

    fn ifetch_at(&mut self, addr: WordType) -> Result<RawInstr, ExceptionInfo> {
        let mut bytes: RawInstr =
            (self
                .read_for_ifetch::<u16>(addr)
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
            let next_half = match self.read_for_ifetch::<u16>(addr + 2) {
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
        self.decode_at_checked(addr).ok()
    }

    #[inline]
    fn decode_at_checked(&mut self, addr: WordType) -> Result<DecodeInstr, ExceptionInfo> {
        if let Some(decoded) = self.icache.get(addr) {
            return Ok(decoded);
        }

        let raw_instr = self.ifetch_at(addr)?;
        let decoded = self.decoder.decode(raw_instr).ok_or(ExceptionInfo {
            cause: Exception::IllegalInstruction,
            tval: raw_instr.val as WordType,
        })?;
        self.icache.put(addr, decoded);
        Ok(decoded)
    }

    fn step_impl(&mut self) {
        let DecodeInstr {
            instr,
            info,
            len: _,
        } = match self.decode_at_checked(self.pc) {
            Ok(decoded) => decoded,
            Err(err) => {
                if err.cause == Exception::IllegalInstruction {
                    cold_path();
                    log::warn!("Illegal instruction: {:#x} at {:#x}", err.tval, self.pc);
                }

                TrapController::take_exception(self, err.cause, err.tval);
                return;
            }
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
