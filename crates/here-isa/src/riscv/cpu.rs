use std::hint::cold_path;

use crate::InstrLen;
use crate::cache::Cache;
use crate::cache::CachePolicy;
use crate::cache::DirectCache;
use crate::cpu::RegFile;
use crate::fpu::soft_float::SoftFPU;
use crate::riscv::RawInstr;
use crate::riscv::csr_reg::{
    CsrDataType, CsrRegfile, Mcycle, Minstret, Mip, Misa, Mstatus, NamedCsr, PrivilegeLevel,
    Sstatus,
};
use crate::riscv::decoder::DecodeInstr;
use crate::riscv::decoder::Decoder;
use crate::riscv::instruction::RVInstrInfo;
use crate::riscv::instruction::exec_mapping::get_exec_func;
use crate::riscv::instruction::instr_table::RiscvInstr;
use crate::riscv::isa_builder::DEFAULT_ISA;
use crate::riscv::isa_builder::ISABuilder;
use crate::riscv::mmu::VirtAddrManager;
use crate::riscv::trap::Exception;
use crate::riscv::trap::trap_controller::TrapController;
use crate::riscv::vector::Vector;
use here_device::irq::RiscvIRQHandler;
use here_types::Interrupt;
use here_types::WordType;
use here_types::ram_config::DEFAULT_PC_VALUE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchResult {
    pub cycles: u64,
    pub hook_stopped: bool,
}

pub trait ExecutorBackend {
    /// Execute exactly `cycles` cycles.
    fn step_batch(&mut self, cpu: &mut RVCPU, cycles: u64);

    /// Prints backend-specific statistics, if any.
    fn report(&self) {}
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
    pub reg_file: RegFile,
    pub pc: WordType,
    pub(crate) memory: VirtAddrManager,
    pub(crate) decoder: Decoder,
    pub csr: CsrRegfile,
    cpl: PrivilegeLevel,
    pub(super) icache: Cache<DirectCache<DecodeInstr, 8192>>,
    icache_epoch: u64,
    pub fpu: SoftFPU,
    pub(crate) vector: Vector,

    /// The address of the memory-mapped `mtime` CSR.
    pub time_addr: Option<WordType>,

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
        Self::from_isa_builder(&DEFAULT_ISA.parse::<ISABuilder>().unwrap(), v_memory)
    }

    pub fn from_isa_builder(isa_builder: &ISABuilder, v_memory: VirtAddrManager) -> Self {
        let mut csr = CsrRegfile::from_builder(&isa_builder);
        let decoder = Decoder::from_builder(&isa_builder);

        let ext = isa_builder.extension_bits();
        csr.context.extension = ext;

        let mxl = if WordType::BITS == 32 {
            1
        } else {
            debug_assert!(WordType::BITS == 64);
            2
        };

        Misa::set_extension_directly(&mut csr, ext as CsrDataType);
        Misa::set_mxl_directly(&mut csr, mxl);
        Mstatus::set_sxl_directly(&mut csr, mxl);
        Mstatus::set_uxl_directly(&mut csr, mxl);

        debug_assert!(Mstatus::get_uxl(&csr) == mxl);
        debug_assert!(Mstatus::get_sxl(&csr) == mxl);
        debug_assert!(Sstatus::get_uxl(&csr) == mxl);

        let fpu = SoftFPU::from(true);

        Self {
            reg_file: RegFile::new(),
            pc: DEFAULT_PC_VALUE,
            memory: v_memory,
            decoder,
            csr,
            cpl: PrivilegeLevel::M,
            vector: Vector::new(),
            icache: Cache::new(),
            icache_epoch: 0,
            fpu,
            time_addr: None,
            pending_tval: None,
        }
    }

    #[inline]
    pub fn privilege_level(&self) -> PrivilegeLevel {
        self.cpl
    }

    #[inline]
    pub fn set_current_privilege(&mut self, new_level: PrivilegeLevel) {
        self.cpl = new_level;
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
    pub fn step_batch_no_interrupt(&mut self, cycles: u64) {
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
        let next = Mcycle::read(&self.csr).wrapping_add(cycles);
        Mcycle::set_mcycle_directly(&mut self.csr, next);
    }

    pub fn try_take_pending_interrupt(&mut self) -> bool {
        if TrapController::try_take_interrupt(self).is_none() {
            return false;
        }

        self.increment_mcycle();
        true
    }

    pub fn finish_jit_block(
        &mut self,
        next_pc: WordType,
        instr_count: u64,
        exception: Option<(Exception, WordType)>,
    ) {
        self.pc = next_pc;
        self.increment_mcycle_by(instr_count + u64::from(exception.is_some()));
        Minstret::wrapping_add(&mut self.csr, instr_count);

        if let Some((exception, tval)) = exception {
            // Memory helpers still call the interpreter-facing read/write APIs, which may leave
            // a compatibility tval behind. JIT exception data comes exclusively from JitContext.
            self.pending_tval = None;
            TrapController::take_exception(self, exception, tval);
        } else {
            debug_assert!(self.pending_tval.is_none());
        }
    }

    pub fn icache_epoch(&self) -> u64 {
        self.icache_epoch
    }

    pub fn instruction_alignment(&mut self) -> WordType {
        if self.csr.has_extension('C') { 2 } else { 4 }
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
    pub fn decode_at(&mut self, addr: WordType) -> Option<DecodeInstr> {
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

    pub fn flush_icache(&mut self) {
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
        let level = level as CsrDataType;

        match interrupt {
            Interrupt::MachineTimer => {
                Mip::set_mtip(&mut self.csr, level);
            }
            Interrupt::MachineExternal => {
                Mip::set_meip(&mut self.csr, level);
            }
            Interrupt::SupervisorExternal => {
                Mip::set_seip(&mut self.csr, level);
            }

            Interrupt::MachineSoft => {
                Mip::set_msip(&mut self.csr, level);
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
