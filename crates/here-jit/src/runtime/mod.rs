use std::hint::cold_path;

use here_isa::cache::*;
#[cfg(any(test, feature = "test-support"))]
use here_isa::riscv::decoder::DecodeInstr;
use here_isa::riscv::executor::ExecutorBackend;
use here_isa::riscv::executor::RVCPU;
use here_types::WordType;

use super::*;

use super::backend::CodeGenTrait;
use super::backend::NativeCodeGen;
use super::backend::PatchLoc;
use super::jit_buffer::JitBuffer;
use super::translator::TranslateResult;
use super::translator::Translator;

pub mod jit_function;

pub use jit_function::JitContext;
pub use jit_function::JitFunction;

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

use super::stats::JitStats;
use rustc_hash::FxHashMap;
use std::fs::File;
use std::io::BufWriter;

#[derive(Debug, Clone, Copy, thiserror::Error)]
pub enum JitError {
    #[error("JIT executor backend is not supported on this platform")]
    Unsupported,
}

pub struct JitRuntime {
    frontend: Translator,
    backend: NativeCodeGen,
    context: Box<JitContext>,
    icache_epoch: u64,
    basic_blocks: BasicBlockManager,
    unresolved_chains: FxHashMap<
        WordType,
        Vec<(
            JitFunction,
            PatchLoc<<NativeCodeGen as CodeGenTrait>::PatchKind>,
        )>,
    >,
    dump_file: Option<BufWriter<File>>,
    pub stats: JitStats,
}

impl JitRuntime {
    pub fn try_new(cpu: &mut RVCPU) -> Result<Self, JitError> {
        let backend = NativeCodeGen::try_new().ok_or(JitError::Unsupported)?;
        let jit_ctx = Box::new(JitContext::default());
        let trans_ctx = translator::TranslatorContext {
            ialign: cpu.instruction_alignment(),
        };

        Ok(Self {
            frontend: Translator::new(trans_ctx),
            backend,
            context: jit_ctx,
            icache_epoch: 0,
            basic_blocks: BasicBlockManager::new(),
            unresolved_chains: FxHashMap::default(),
            dump_file: None,
            stats: JitStats::default(),
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

    fn compile(&mut self, cpu: &mut RVCPU, pc: WordType) -> Option<BasicBlock> {
        let TranslateResult {
            ir_block,
            meta,
            guest_bytes,
            stop_reason,
        } = match self.frontend.translate(cpu, pc) {
            Ok(res) => res,
            Err(stop_reason) => {
                self.stats.record_unsupported_first_instr(stop_reason);
                return None;
            }
        };

        let cb = self.backend.compile(&ir_block, &meta);
        self.stats.record_compile(
            meta.guest_cnt as u64,
            guest_bytes,
            cb.buf.len(),
            stop_reason,
        );

        if let Some(file) = &mut self.dump_file {
            write_block_dump(
                file,
                &format!(
                    "Block guest PC: {:#018x}, instructions: {}",
                    pc, meta.guest_cnt
                ),
                &ir_block,
                cb.asm_text,
            );
        }

        let func = unsafe { self.basic_blocks.emit_code(cb.buf) };

        // 1. Resolve any previous blocks waiting to chain into this pc:
        if let Some(waiting) = self.unresolved_chains.remove(&pc) {
            self.stats.record_chain_forward_resolved(waiting.len());
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
                self.stats.record_chain_immediate();
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
                self.stats.record_chain_immediate();
                unsafe {
                    NativeCodeGen::patch(
                        func.as_mut_ptr(),
                        patch.loc,
                        target_bb.func.as_ptr() as usize as i64,
                    );
                }
            } else {
                self.stats.record_chain_forward_queued();
                self.unresolved_chains
                    .entry(patch.target_pc)
                    .or_default()
                    .push((func, patch.loc));
            }
        }

        Some(BasicBlock { func })
    }

    fn get_or_compile(&mut self, cpu: &mut RVCPU, pc: WordType) -> BBCacheEntry {
        if let Some(entry) = self.basic_blocks.try_get(pc) {
            return entry;
        }
        let entry = match self.compile(cpu, pc) {
            Some(bb) => BBCacheEntry::Compiled(bb),
            None => BBCacheEntry::InterpreterOnly,
        };
        self.basic_blocks.put(pc, entry);
        entry
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn try_execute_instruction_for_test(
        &mut self,
        cpu: &mut RVCPU,
        decoded: DecodeInstr,
    ) -> bool {
        let Some((ir_block, meta)) = self.frontend.translate_single_instruction(cpu.pc, decoded)
        else {
            return false;
        };

        let cb = self.backend.compile(&ir_block, &meta);
        if let Some(file) = &mut self.dump_file {
            write_block_dump(
                file,
                &format!("Test Instruction guest PC: {:#018x}", cpu.pc),
                &ir_block,
                cb.asm_text,
            );
        }
        let func = unsafe { self.basic_blocks.emit_code(cb.buf) };
        let bb = BasicBlock { func };
        self.context.budget = 1;
        self.run_jit_block(cpu, bb);
        true
    }
}

fn write_block_dump(
    file: &mut BufWriter<File>,
    title: &str,
    ir_block: &IRBlock,
    asm_text: Option<&str>,
) {
    use std::io::Write;

    let _ = writeln!(
        file,
        "# =============================================================================="
    );
    let _ = writeln!(file, "# === {title} ===");
    let _ = writeln!(
        file,
        "# =============================================================================="
    );
    let _ = writeln!(file, "# --- IR ---");
    let _ = writeln!(file, "{ir_block}");
    let _ = writeln!(file, "# --- Host ASM ---");
    if let Some(asm) = asm_text {
        let _ = file.write_all(asm.as_bytes());
    }
    let _ = writeln!(file);
    let _ = file.flush();
}

struct BasicBlockManager {
    // BB cache must never evict or have at least 99.5% hit rate,
    // because eviction drops JIT function pointers,
    // leaving old translated blocks forgotten and causing high JIT memory usage.
    cache: Cache<ChainedCache<DirectCache<BBCacheEntry, 4096>, FxCache<BBCacheEntry>>>,
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

    fn try_get(&mut self, pc: WordType) -> Option<BBCacheEntry> {
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

    fn buffer_allocated_bytes(&self) -> usize {
        self.jit_buffer.latest_allocated_bytes()
    }

    fn buffer_used_bytes(&self) -> usize {
        self.jit_buffer.latest_used_bytes()
    }

    fn buffer_peak_bytes(&self) -> usize {
        self.jit_buffer.peak_used_bytes()
    }
}

impl JitRuntime {
    fn sync_icache_epoch(&mut self, cpu: &RVCPU) {
        let epoch = cpu.icache_epoch();
        if epoch == self.icache_epoch {
            return;
        }

        cold_path();
        self.stats.record_epoch_invalidation();
        self.basic_blocks.clear();
        self.unresolved_chains.clear();
        self.icache_epoch = epoch;
    }

    fn consume_cycles(&mut self, cycles: u64) {
        self.context.budget -= cycles;
    }

    fn run_interpreter(&mut self, cpu: &mut RVCPU, cycles: u64) {
        self.stats.record_interpreter_cycles(cycles);
        cpu.step_batch_no_interrupt(cycles);
        self.consume_cycles(cycles);
    }

    fn run_jit_block(&mut self, cpu: &mut RVCPU, mut bb: BasicBlock) {
        self.stats.record_jit_invocation();
        let initial_budget = self.context.budget;
        let exit = unsafe { bb.func.call(cpu, self.context.as_mut()) };
        if exit.has_exception {
            self.handle_exception_exit(cpu, initial_budget);
            return;
        }
        let consumed = initial_budget - self.context.budget;
        if consumed == 0 {
            self.stats.record_out_of_budget();
            self.run_interpreter(cpu, self.context.budget);
            return;
        }
        self.stats.record_jit_cycles(consumed);
        cpu.finish_jit_block(exit.next_pc, consumed, None);
    }

    #[cold]
    fn handle_exception_exit(&mut self, cpu: &mut RVCPU, initial_budget: u64) {
        self.stats.record_exception_exit();
        let (exception, tval) = self
            .context
            .exception
            .take()
            .expect("a JIT abort must record an exception");
        let consumed = initial_budget - self.context.budget;
        let instr_count = consumed - 1;
        self.stats.record_jit_cycles(instr_count);
        cpu.finish_jit_block(self.context.guest_pc, instr_count, Some((exception, tval)));
    }

    pub fn step_batch(&mut self, cpu: &mut RVCPU, cycles: u64) {
        if cycles == 0 {
            return;
        }

        debug_assert_eq!(self.context.exception, None);
        self.context.budget = cycles;

        if cpu.try_take_pending_interrupt() {
            self.stats.record_interpreter_cycles(1);
            self.consume_cycles(1);
        }

        while self.context.budget > 0 {
            self.sync_icache_epoch(cpu);

            match self.get_or_compile(cpu, cpu.pc) {
                BBCacheEntry::Compiled(bb) => {
                    self.run_jit_block(cpu, bb);
                }
                BBCacheEntry::InterpreterOnly => {
                    self.run_interpreter(cpu, 1);
                    continue;
                }
            }
        }
    }

    /// Prints collected statistics.
    pub fn report(&self) {
        if !self.stats.has_activity() {
            return;
        }
        let report = self.stats.format_report(
            self.basic_blocks.hit_count(),
            self.basic_blocks.access_count(),
            self.basic_blocks.buffer_allocated_bytes(),
            self.basic_blocks.buffer_used_bytes(),
            self.basic_blocks.buffer_peak_bytes(),
        );
        println!("\n{}", report);
    }
}

impl ExecutorBackend for JitRuntime {
    fn step_batch(&mut self, cpu: &mut RVCPU, cycles: u64) {
        JitRuntime::step_batch(self, cpu, cycles);
    }

    fn report(&self) {
        JitRuntime::report(self);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::ir::IRBuilder;
    use crate::make_ir;
    use here_isa::DebugTarget;
    use here_isa::riscv::address::Address;
    use here_isa::riscv::cpu_tester::TestCPUBuilder;
    use here_isa::riscv::csr_reg::NamedCsrReg;
    use here_isa::riscv::csr_reg::csr_macro::Mcycle;
    use here_isa::riscv::csr_reg::csr_macro::Minstret;
    use here_isa::riscv::trap::Exception;
    use here_types::ram_config::BASE_ADDR;

    #[test]
    fn test_block_chaining_forward_and_reexecution() {
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

        let Ok(mut jit) = JitRuntime::try_new(&mut cpu) else {
            return;
        };
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

        let Ok(mut jit) = JitRuntime::try_new(&mut cpu) else {
            return;
        };
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

        let Ok(mut jit) = JitRuntime::try_new(&mut cpu) else {
            return;
        };
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

        let Ok(mut jit) = JitRuntime::try_new(&mut cpu) else {
            return;
        };
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

        let Ok(mut jit) = JitRuntime::try_new(&mut cpu) else {
            return;
        };

        // Compile and chain Block 1 -> Block 2:
        jit.step_batch(&mut cpu, 4);
        assert_eq!(cpu.read_reg(1), 10);
        assert_eq!(cpu.read_reg(2), 20);

        // Modify Block 2 in memory: change addi x2, x2, 20 to addi x2, x2, 100
        cpu.write_memory(Address::Virt(BASE_ADDR + 8), 0x0641_0113_u32)
            .unwrap();
        // Invalidate icache epoch:
        cpu.flush_icache();
        // Reset PC back to Block 1 and run again:
        cpu.pc = BASE_ADDR;
        jit.step_batch(&mut cpu, 4);

        assert_eq!(cpu.read_reg(1), 20);
        // Must reflect modified Block 2 instruction (+100 instead of +20)
        assert_eq!(cpu.read_reg(2), 120);
    }

    #[test]
    fn test_jit_stats_execution() {
        let mut cpu = TestCPUBuilder::new()
            .pc(BASE_ADDR)
            .mem_base(0, 0x00a0_8093_u32) // addi x1, x1, 10
            .mem_base(4, 0x0040_006f_u32) // jal x0, 4 (to BASE_ADDR + 8)
            .mem_base(8, 0x0000_006f_u32) // jal x0, 0
            .build();

        let Ok(mut jit) = JitRuntime::try_new(&mut cpu) else {
            return;
        };
        jit.step_batch(&mut cpu, 2);
        assert!(jit.stats.compiled_blocks >= 1);
        assert!(jit.stats.compiled_guest_insts >= 2);
        assert!(jit.stats.compiled_guest_bytes >= 8);
        assert_eq!(jit.stats.jit_cycles, 2);
        assert_eq!(jit.stats.jit_block_invocations, 1);
        assert!(jit.stats.has_activity());
        let report = jit.stats.format_report(1, 1, 2 * 1024 * 1024, 64, 64);
        assert!(report.contains("JIT Executed Cycles"));
        assert!(report.contains("Compiled Basic Blocks"));
    }

    #[test]
    fn test_jit_exception_trampoline_abort() {
        unsafe extern "sysv64" fn helper_abort(ctx: *mut JitContext) {
            unsafe {
                JitContext::raise(
                    ctx,
                    0x1234_5678,
                    5,
                    Exception::InstructionMisaligned,
                    0x1234_5679,
                );
            }
        }

        let mut cpu = TestCPUBuilder::new().build();
        let mut ctx = JitContext::default();
        ctx.budget = 10;

        let mut builder = IRBuilder::default();
        let ctx_vreg = builder.jit_context_vreg();
        make_ir!(builder, [
            v0 = const 100;
            set_reg 1, v0;
            call [trap] (helper_abort as *const () as usize), [ctx_vreg];
            v1 = const 200;
            set_reg 1, v1;
        ], []);
        let block = builder.build_jmp(0x2000u64.into());

        let exit = crate::backend::tests::compile_and_exec_with_ctx(
            &block,
            &IRBlockMeta::default(),
            cpu.as_mut(),
            &mut ctx,
        );

        assert!(exit.has_exception);
        assert_eq!(exit.next_pc, 0);
        assert_eq!(ctx.guest_pc, 0x1234_5678);
        assert_eq!(ctx.budget, 5);
        assert_eq!(
            ctx.exception,
            Some((Exception::InstructionMisaligned, 0x1234_5679))
        );
        assert_eq!(cpu.read_reg(1), 100);

        // Verify subsequent JIT execution continues cleanly:
        let mut builder2 = IRBuilder::default();
        make_ir!(builder2, [
            v2 = const 300;
            set_reg 2, v2;
        ], []);
        let block2 = builder2.build_jmp(0x3000u64.into());
        let exit2 = crate::backend::tests::compile_and_exec_with_ctx(
            &block2,
            &IRBlockMeta::default(),
            cpu.as_mut(),
            &mut ctx,
        );
        assert!(!exit2.has_exception);
        assert_eq!(exit2.next_pc, 0x3000);
        assert_eq!(cpu.read_reg(2), 300);
    }
}
