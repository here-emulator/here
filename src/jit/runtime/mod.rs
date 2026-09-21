use crate::config::arch_config::WordType;
use crate::isa::cache::*;
use crate::isa::riscv::executor::{ExecutorBackend, RVCPU};
use crate::jit::new_translator::TranslateResult;
mod jit_function;

pub use jit_function::{JitContext, JitFunction};

use super::*;

use super::backend::{CodeGenTrait, NativeCodeGen};
use super::jit_buffer::JitBuffer;
use super::new_translator::Translator;

impl Cacheable for BBCacheEntry {
    const ADDR_SHIFT_BITS: usize = 1;
}

#[derive(Clone, Copy)]
struct BasicBlock {
    max_instr_cnt: u64,
    func: JitFunction,
}

#[derive(Clone, Copy)]
enum BBCacheEntry {
    Compiled(BasicBlock),
    InterpreterOnly,
}

use std::fs::File;
use std::io::BufWriter;

pub struct JitRuntime {
    frontend: Translator,
    backend: NativeCodeGen,
    context: Box<JitContext>,
    icache_epoch: u64,
    basic_blocks: BasicBlockManager,
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

        Some(BasicBlock {
            max_instr_cnt: instr_cnt,
            func,
        })
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
        let bb = BasicBlock {
            max_instr_cnt: 1,
            func,
        };
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
        debug_assert!(bb.max_instr_cnt <= self.context.budget);

        let next_pc = unsafe { bb.func.call(cpu as *mut RVCPU) };
        cpu.finish_jit_block(next_pc, bb.max_instr_cnt, None);
        if self.context.budget == self.block_initial_budget {
            self.consume_cycles(bb.max_instr_cnt);
        }
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
                    let remaining_cycles = self.context.budget;
                    if bb.max_instr_cnt > remaining_cycles {
                        self.run_interpreter(cpu, remaining_cycles);
                        break;
                    }
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
