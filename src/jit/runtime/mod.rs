use crate::config::arch_config::WordType;
use crate::isa::cache::*;
use crate::isa::riscv::executor::{ExecutorBackend, RVCPU};

use super::*;

use super::backend::{CodeGenTrait, NativeCodeGen, PatchLoc};
use super::jit_buffer::JitBuffer;
use super::new_translator::{TranslateResult, Translator};

mod jit_function;

pub use jit_function::{JitContext, JitFunction};

impl Cacheable for BBCacheEntry {
    const ADDR_SHIFT_BITS: usize = 1;
}

#[derive(Clone, Copy)]
struct BasicBlock {
    func: JitFunction,
}

#[derive(Clone, Copy)]
enum BBCacheEntry {
    Compiled(BasicBlock),
    InterpreterOnly,
}

use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;

pub struct JitRuntime {
    frontend: Translator,
    backend: NativeCodeGen,
    context: Box<JitContext>,
    icache_epoch: u64,
    basic_blocks: BasicBlockManager,
    unresolved_chains: HashMap<
        WordType,
        Vec<(
            JitFunction,
            PatchLoc<<NativeCodeGen as CodeGenTrait>::PatchKind>,
        )>,
    >,
    // TODO: remove with a local variable when longjmp/setjmp is replaced
    block_initial_budget: u64,
    dump_file: Option<BufWriter<File>>,
}

impl JitRuntime {
    #[inline]
    pub fn is_supported() -> bool {
        NativeCodeGen::try_new().is_some()
    }

    // TODO: Error instead of Option
    pub fn try_new() -> Option<Self> {
        let mut jit_ctx = Box::new(JitContext::default());
        let trans_ctx = new_translator::TranslatorContext {
            jit_context: jit_ctx.as_mut(),
        };

        Some(Self {
            frontend: Translator::new(trans_ctx),
            backend: NativeCodeGen::try_new()?,
            context: jit_ctx,
            icache_epoch: 0,
            basic_blocks: BasicBlockManager::new(),
            unresolved_chains: HashMap::new(),
            block_initial_budget: 0,
            dump_file: None,
        })
    }

    pub fn enable_dump(&mut self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = File::create(path)?;
        self.backend.set_dump_asm(true);
        self.dump_file = Some(BufWriter::new(file));
        Ok(())
    }

    pub fn with_dump(mut self, path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        self.enable_dump(path)?;
        Ok(self)
    }

    pub fn set_dump_writer(&mut self, writer: BufWriter<File>) {
        self.backend.set_dump_asm(true);
        self.dump_file = Some(writer);
    }

    pub fn disable_dump(&mut self) {
        self.backend.set_dump_asm(false);
        self.dump_file = None;
    }

    fn compile(&mut self, cpu: &mut RVCPU, pc: WordType) -> Option<BasicBlock> {
        let TranslateResult {
            ir_block,
            instr_cnt,
        } = self.frontend.translate(cpu, pc)?;

        let cb = self.backend.compile(&ir_block);

        if let Some(ref mut file) = self.dump_file {
            use std::io::Write;
            let _ = writeln!(
                file,
                "; =============================================================================="
            );
            let _ = writeln!(
                file,
                "; === Block guest PC: {:#018x}, instructions: {} ===",
                pc, instr_cnt
            );
            let _ = writeln!(
                file,
                "; =============================================================================="
            );
            let _ = writeln!(file, "; --- IR ---");
            let _ = writeln!(file, "{}", ir_block);
            let _ = writeln!(file, "; --- Host ASM ---");
            if let Some(asm) = &cb.asm_text {
                let _ = file.write_all(asm.as_bytes());
                if !asm.ends_with('\n') {
                    let _ = writeln!(file);
                }
            }
            let _ = writeln!(file);
            let _ = file.flush();
        }

        let func = unsafe { self.basic_blocks.emit_code(cb.buf) };

        // 1. Resolve any previous blocks waiting to chain into this pc:
        if let Some(waiting) = self.unresolved_chains.remove(&pc) {
            for (src_func, patch) in waiting {
                unsafe {
                    NativeCodeGen::patch(
                        src_func.as_mut_ptr(),
                        patch,
                        func.as_ptr() as usize as i64,
                    );
                }
            }
        }

        // 2. Chain exits from this block:
        for patch in cb.patches {
            if patch.target_pc == pc {
                unsafe {
                    NativeCodeGen::patch(
                        func.as_mut_ptr(),
                        patch.loc,
                        func.as_ptr() as usize as i64,
                    );
                }
            } else if let Some(BBCacheEntry::Compiled(target_bb)) =
                self.basic_blocks.try_get(patch.target_pc)
            {
                unsafe {
                    NativeCodeGen::patch(
                        func.as_mut_ptr(),
                        patch.loc,
                        target_bb.func.as_ptr() as usize as i64,
                    );
                }
            } else {
                self.unresolved_chains
                    .entry(patch.target_pc)
                    .or_default()
                    .push((func, patch.loc));
            }
        }

        Some(BasicBlock { func })
    }

    fn get_or_compile(&mut self, cpu: &mut RVCPU, pc: WordType) -> BBCacheEntry {
        self.basic_blocks.try_get(pc).unwrap_or_else(|| {
            let entry = match self.compile(cpu, pc) {
                Some(bb) => BBCacheEntry::Compiled(bb),
                None => BBCacheEntry::InterpreterOnly,
            };
            self.basic_blocks.put(pc, entry);
            entry
        })
    }

    #[cfg(test)]
    #[inline(never)]
    pub fn try_execute_instruction_for_test(
        &mut self,
        cpu: &mut RVCPU,
        instr: crate::isa::riscv::instruction::instr_table::RiscvInstr,
        info: crate::isa::riscv::instruction::RVInstrInfo,
    ) -> bool {
        let Some(ir_block) = self
            .frontend
            .translate_single_instruction(cpu.pc, instr, info)
        else {
            return false;
        };

        let cb = self.backend.compile(&ir_block);
        if let Some(ref mut file) = self.dump_file {
            use std::io::Write;
            let _ = writeln!(
                file,
                "; =============================================================================="
            );
            let _ = writeln!(
                file,
                "; === Test Instruction guest PC: {:#018x} ===",
                cpu.pc
            );
            let _ = writeln!(
                file,
                "; =============================================================================="
            );
            let _ = writeln!(file, "; --- IR ---");
            let _ = writeln!(file, "{}", ir_block);
            let _ = writeln!(file, "; --- Host ASM ---");
            if let Some(asm) = &cb.asm_text {
                let _ = file.write_all(asm.as_bytes());
                if !asm.ends_with('\n') {
                    let _ = writeln!(file);
                }
            }
            let _ = writeln!(file);
            let _ = file.flush();
        }
        let func = unsafe { self.basic_blocks.emit_code(cb.buf) };
        let bb = BasicBlock { func };
        self.context.budget = 1;
        self.block_initial_budget = 1;
        if unsafe { setjmp(self.context.jmp_buf_ptr()) } != 0 {
            self.handle_exception_exit(cpu);
        } else {
            self.run_jit_block(cpu, bb);
        }
        true
    }
}

struct BasicBlockManager {
    // TODO: BB cache eviction drops JIT function pointers,
    // leaving old translated blocks unused and causing high JIT memory usage.
    // `SetCache` is used to improve hit rates and ease this issue.
    // - use a quick hashmap (maybe `rustc_hash::FxHashMap`) for secondary lookup
    // - consider if we should reset jit buffer if its too large
    cache: Cache<SetCache<BBCacheEntry, 4096, 8>>,
    jit_buffer: JitBuffer,
}

impl BasicBlockManager {
    fn new() -> Self {
        Self {
            cache: Cache::new(),
            jit_buffer: JitBuffer::new(),
        }
    }

    fn clear(&mut self) {
        self.cache.clear();
        self.jit_buffer.reset();
    }

    fn try_get(&self, pc: WordType) -> Option<BBCacheEntry> {
        self.cache.get(pc)
    }

    fn put(&mut self, pc: WordType, entry: BBCacheEntry) {
        self.cache.put(pc, entry);
    }

    unsafe fn emit_code(&mut self, code: &[u8]) -> JitFunction {
        unsafe { self.jit_buffer.emit_code(code) }
    }

    fn hit_count(&self) -> u64 {
        self.cache.hit_count()
    }

    fn access_count(&self) -> u64 {
        self.cache.access_count()
    }
}

// TODO: migrate from legacy code, can be re-written

use std::hint::cold_path;

use libc::{c_int, c_void};

unsafe extern "C" {
    /// XXX: Make sure you know what you're doing.
    ///
    /// rustc doesn't support wired function that "return twice",
    /// so calling this is actually an UB, and llvm cannot recognize this hidden control flow,
    /// handle it carefully:
    ///
    /// - Use `#[inline(never)]` for every function that calls `setjmp`.
    /// - After `setjmp`, do not use local mutable variable that defined before.
    /// - Check the local variable lifetime so that stack-slot reuse won't break anything.
    fn setjmp(env: *mut c_void) -> c_int;
}

impl JitRuntime {
    fn sync_icache_epoch(&mut self, cpu: &RVCPU) {
        let epoch = cpu.icache_epoch();
        if epoch == self.icache_epoch {
            return;
        }

        cold_path();
        self.basic_blocks.clear();
        self.unresolved_chains.clear();
        self.icache_epoch = epoch;
    }

    fn consume_cycles(&mut self, cycles: u64) {
        self.context.budget -= cycles;
    }

    fn run_interpreter(&mut self, cpu: &mut RVCPU, cycles: u64) {
        cpu.step_batch_no_interrupt(cycles);
        self.consume_cycles(cycles);
    }

    fn run_jit_block(&mut self, cpu: &mut RVCPU, bb: BasicBlock) {
        let next_pc = unsafe {
            bb.func
                .call(cpu as *mut RVCPU, self.context.as_mut() as *mut JitContext)
        };
        let consumed = self.block_initial_budget - self.context.budget;
        if consumed == 0 {
            let remaining = self.context.budget;
            self.run_interpreter(cpu, remaining);
            return;
        }
        cpu.finish_jit_block(next_pc, consumed, None);
    }

    fn handle_exception_exit(&mut self, cpu: &mut RVCPU) {
        let (exception, tval) = self
            .context
            .exception
            .take()
            .expect("a JIT longjmp must record an exception");
        let consumed = self.block_initial_budget - self.context.budget;
        let instr_count = consumed - 1;
        cpu.finish_jit_block(self.context.guest_pc, instr_count, Some((exception, tval)));
    }
}

impl ExecutorBackend for JitRuntime {
    #[inline(never)] // See `setjmp`
    fn step_batch(&mut self, cpu: &mut RVCPU, cycles: u64) {
        if cycles == 0 {
            return;
        }

        debug_assert_eq!(self.context.exception, None);
        self.context.budget = cycles;

        // Don't use `cycles` directly after this because longjmp may break the value.
        #[allow(unused)]
        let cycles = "DO NOT USE";

        if cpu.try_take_pending_interrupt() {
            self.consume_cycles(1);
        }

        if unsafe { setjmp(self.context.jmp_buf_ptr()) != 0 } {
            self.handle_exception_exit(cpu);
            // fall through, run remaining cycles.
        }

        while self.context.budget > 0 {
            self.sync_icache_epoch(cpu);

            match self.get_or_compile(cpu, cpu.pc) {
                BBCacheEntry::Compiled(bb) => {
                    self.block_initial_budget = self.context.budget;
                    self.run_jit_block(cpu, bb);
                }
                BBCacheEntry::InterpreterOnly => {
                    self.run_interpreter(cpu, 1);
                    continue;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::DebugTarget;
    use crate::isa::riscv::cpu_tester::TestCPUBuilder;
    use crate::isa::riscv::csr_reg::NamedCsrReg;
    use crate::isa::riscv::csr_reg::csr_macro::{Mcycle, Minstret, Satp};
    use crate::isa::riscv::debugger::Address;
    use crate::ram_config::BASE_ADDR;

    #[test]
    fn test_block_chaining_forward_and_reexecution() {
        if !JitRuntime::is_supported() {
            return;
        }

        // Block 1 (BASE_ADDR):
        //   addi x1, x1, 10 (0x00a08093)
        //   jal x0, 4       (0x0040006f) -> jumps to BASE_ADDR + 8 (Block 2)
        // Block 2 (BASE_ADDR + 8):
        //   addi x2, x2, 20 (0x01410113)
        //   jal x0, 4       (0x0040006f) -> jumps to BASE_ADDR + 16 (Block 3)
        // Block 3 (BASE_ADDR + 16):
        //   addi x3, x3, 30 (0x01e18193)
        //   jal x0, 0       (0x0000006f) -> self-loop
        let mut cpu = TestCPUBuilder::new()
            .pc(BASE_ADDR)
            .mem_base(0, 0x00a0_8093_u32)
            .mem_base(4, 0x0040_006f_u32)
            .mem_base(8, 0x0141_0113_u32)
            .mem_base(12, 0x0040_006f_u32)
            .mem_base(16, 0x01e1_8193_u32)
            .mem_base(20, 0x0000_006f_u32)
            .build();

        let mut jit = JitRuntime::try_new().unwrap();

        // Run 4 cycles: executes Block 1 (2 insts) and Block 2 (2 insts).
        // During compilation of Block 2, Block 1's forward jump is patched (resolved).
        jit.step_batch(&mut cpu, 4);
        assert_eq!(cpu.pc, BASE_ADDR + 16);
        assert_eq!(cpu.read_reg(1), 10);
        assert_eq!(cpu.read_reg(2), 20);
        assert_eq!(cpu.read_reg(3), 0);
        assert_eq!(cpu.debug_csr(Mcycle::get_index(), None), Some(4));
        assert_eq!(cpu.debug_csr(Minstret::get_index(), None), Some(4));

        // Reset PC back to Block 1. Both blocks are now compiled and chained!
        cpu.pc = BASE_ADDR;
        // Running 4 cycles will jump directly from Block 1 into Block 2 in host machine code.
        jit.step_batch(&mut cpu, 4);
        assert_eq!(cpu.pc, BASE_ADDR + 16);
        assert_eq!(cpu.read_reg(1), 20);
        assert_eq!(cpu.read_reg(2), 40);
        assert_eq!(cpu.read_reg(3), 0);
        assert_eq!(cpu.debug_csr(Mcycle::get_index(), None), Some(8));
        assert_eq!(cpu.debug_csr(Minstret::get_index(), None), Some(8));
    }

    #[test]
    fn test_block_chaining_backward_compile() {
        if !JitRuntime::is_supported() {
            return;
        }

        // Block 1 (BASE_ADDR):
        //   addi x1, x1, 10 (0x00a08093)
        //   jal x0, 4       (0x0040006f) -> jumps to BASE_ADDR + 8 (Block 2)
        // Block 2 (BASE_ADDR + 8):
        //   addi x2, x2, 20 (0x01410113)
        //   jal x0, 0       (0x0000006f) -> self-loop
        let mut cpu = TestCPUBuilder::new()
            .pc(BASE_ADDR + 8)
            .mem_base(0, 0x00a0_8093_u32)
            .mem_base(4, 0x0040_006f_u32)
            .mem_base(8, 0x0141_0113_u32)
            .mem_base(12, 0x0000_006f_u32)
            .build();

        let mut jit = JitRuntime::try_new().unwrap();

        // Compile and run Block 2 first:
        jit.step_batch(&mut cpu, 2);
        assert_eq!(cpu.read_reg(2), 20);

        // Now point PC to Block 1. When Block 1 compiles, Block 2 is ALREADY in the BB cache.
        // Block 1 will immediately patch its target to Block 2.
        cpu.pc = BASE_ADDR;
        jit.step_batch(&mut cpu, 4);
        assert_eq!(cpu.read_reg(1), 10);
        assert_eq!(cpu.read_reg(2), 40);
    }

    #[test]
    fn test_block_chaining_self_loop() {
        if !JitRuntime::is_supported() {
            return;
        }

        // Tight loop:
        // BASE_ADDR:
        //   addi x1, x1, -1 (0xfff08093)
        //   bne  x1, x0, -4 (0xfe009ee3) -> jump to BASE_ADDR
        // BASE_ADDR + 8:
        //   addi x2, x2, 100 (0x06400113)
        //   jal  x0, 0       (0x0000006f)
        let mut cpu = TestCPUBuilder::new()
            .pc(BASE_ADDR)
            .reg(1, 50)
            .mem_base(0, 0xfff0_8093_u32)
            .mem_base(4, 0xfe00_9ee3_u32)
            .mem_base(8, 0x0640_0113_u32)
            .mem_base(12, 0x0000_006f_u32)
            .build();

        let mut jit = JitRuntime::try_new().unwrap();

        // 50 iterations * 2 insts + 1 fallthrough inst (addi x2) + 1 (jal x0, 0) = 102 cycles
        jit.step_batch(&mut cpu, 102);

        assert_eq!(cpu.read_reg(1), 0);
        assert_eq!(cpu.read_reg(2), 100);
        assert_eq!(cpu.pc, BASE_ADDR + 12);
        assert_eq!(cpu.debug_csr(Mcycle::get_index(), None), Some(102));
        assert_eq!(cpu.debug_csr(Minstret::get_index(), None), Some(102));
    }

    #[test]
    fn test_block_chaining_loop_budget_exhaustion_and_cold_path() {
        if !JitRuntime::is_supported() {
            return;
        }

        // Loop:
        // BASE_ADDR:
        //   addi x1, x1, -1 (0xfff08093)
        //   bne  x1, x0, -4 (0xfe009ee3)
        let mut cpu = TestCPUBuilder::new()
            .pc(BASE_ADDR)
            .reg(1, 50)
            .mem_base(0, 0xfff0_8093_u32)
            .mem_base(4, 0xfe00_9ee3_u32)
            .build();

        let mut jit = JitRuntime::try_new().unwrap();

        // Run with budget = 25 cycles (odd number of cycles).
        // 12 iterations in JIT = 24 instructions.
        // Remaining budget = 1, insufficient for 2-inst block.
        // Out-of-budget exit triggers, consumed = 0, falls back to interpreter for 1 cycle.
        // Total instructions executed = 25 (12 * 2 + 1 = 25).
        jit.step_batch(&mut cpu, 25);

        // x1 decremented 13 times (12 times in JIT, 1 time in interpreter): 50 - 13 = 37.
        assert_eq!(cpu.read_reg(1), 37);
        // PC after interpreter runs addi at BASE_ADDR is BASE_ADDR + 4.
        assert_eq!(cpu.pc, BASE_ADDR + 4);
        assert_eq!(cpu.debug_csr(Mcycle::get_index(), None), Some(25));
        assert_eq!(cpu.debug_csr(Minstret::get_index(), None), Some(25));
    }

    #[test]
    fn test_icache_epoch_invalidation_clears_chains() {
        if !JitRuntime::is_supported() {
            return;
        }

        // Initial program:
        // Block 1: addi x1, x1, 10; jal x0, 4
        // Block 2: addi x2, x2, 20; jal x0, 0
        let mut cpu = TestCPUBuilder::new()
            .pc(BASE_ADDR)
            .mem_base(0, 0x00a0_8093_u32)
            .mem_base(4, 0x0040_006f_u32)
            .mem_base(8, 0x0141_0113_u32)
            .mem_base(12, 0x0000_006f_u32)
            .build();

        let mut jit = JitRuntime::try_new().unwrap();

        // Compile and chain Block 1 -> Block 2:
        jit.step_batch(&mut cpu, 4);
        assert_eq!(cpu.read_reg(1), 10);
        assert_eq!(cpu.read_reg(2), 20);

        // Modify Block 2 in memory: change addi x2, x2, 20 to addi x2, x2, 100
        cpu.write_memory(Address::Virt(BASE_ADDR + 8), 0x0641_0113_u32)
            .unwrap();
        // Invalidate icache epoch:
        cpu.write_csr(Satp::get_index(), 1).unwrap();

        // Reset PC back to Block 1 and run again:
        cpu.pc = BASE_ADDR;
        jit.step_batch(&mut cpu, 4);

        assert_eq!(cpu.read_reg(1), 20);
        // Must reflect modified Block 2 instruction (+100 instead of +20)
        assert_eq!(cpu.read_reg(2), 120);
    }
}
