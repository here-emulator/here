#[cfg(feature = "jit-stats")]
use std::collections::BTreeMap;

use crate::isa::riscv::instruction::instr_table::RiscvInstr;
#[cfg(feature = "jit-stats")]
use crate::jit::translator::MAX_BLOCK_INSTRUCTIONS;

/// Reason why translation of a basic block terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockStopReason {
    /// Conditional branch instruction (e.g. beq, bne, blt, bge).
    Branch(RiscvInstr),
    /// Direct unconditional jump (e.g. jal, c.j).
    DirectJump(RiscvInstr),
    /// Dynamic/indirect jump (e.g. jalr, c.jr, c.jalr).
    DynamicJump(RiscvInstr),
    /// Current JIT does not support this instruction; falls back to interpreter.
    Unsupported(RiscvInstr),
    /// Translation stopped because the maximum block length limit was reached.
    BlockLimit,
    /// Decoding failed (e.g. MMU fetch fault or invalid instruction).
    DecodeFault,
}

#[inline]
fn safe_ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator > 0 {
        numerator as f64 / denominator as f64
    } else {
        0.0
    }
}

#[inline]
fn safe_pct(part: u64, total: u64) -> f64 {
    safe_ratio(part, total) * 100.0
}

#[inline]
fn bytes_to_kb(bytes: u64) -> f64 {
    bytes as f64 / 1024.0
}

#[inline]
fn bytes_to_mb(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

#[cfg(feature = "jit-stats")]
#[derive(Default, Debug, Clone)]
pub struct JitStats {
    // --- Execution & Coverage ---
    pub jit_cycles: u64,
    pub interpreter_cycles: u64,
    pub jit_block_invocations: u64,
    pub out_of_budget_exits: u64,
    pub exception_exits: u64,

    // --- Compilation Quality ---
    pub compiled_blocks: u64,
    pub compiled_guest_insts: u64,
    pub compiled_guest_bytes: u64,
    pub host_code_bytes: u64,

    // --- Block Stop Reasons ---
    pub direct_jumps: u64,
    pub branches: u64,
    pub dynamic_jumps: u64,
    pub block_limits: u64,
    pub decode_faults: u64,
    pub unsupported_insts: BTreeMap<RiscvInstr, u64>,
    pub total_unsupported: u64,

    // --- Chaining & Epochs ---
    pub chained_immediate: u64,
    pub unresolved_chains_queued: u64,
    pub unresolved_chains_resolved: u64,
    pub epoch_invalidations: u64,
}

#[cfg(feature = "jit-stats")]
struct ReportWriter {
    buf: String,
}

#[cfg(feature = "jit-stats")]
impl ReportWriter {
    fn new() -> Self {
        Self {
            buf: String::with_capacity(2048),
        }
    }

    fn line(&mut self, text: &str) {
        self.buf.push_str(text);
        self.buf.push('\n');
    }

    fn section(&mut self, title: &str) {
        self.buf.push_str("\r\n[");
        self.buf.push_str(title);
        self.buf.push_str("]\n");
    }

    fn field_u64(&mut self, label: &str, val: u64) {
        use std::fmt::Write;
        let _ = writeln!(self.buf, "  {:<29} {:>12}", label, val);
    }

    fn field_pct(&mut self, label: &str, val: u64, pct: f64) {
        use std::fmt::Write;
        let _ = writeln!(self.buf, "  {:<29} {:>12} ({:6.2}%)", label, val, pct);
    }

    fn field_pct_suffix(&mut self, label: &str, val: u64, pct: f64, suffix: &str) {
        use std::fmt::Write;
        let _ = writeln!(
            self.buf,
            "  {:<29} {:>12} ({:6.2}% {})",
            label, val, pct, suffix
        );
    }

    fn field_f64(&mut self, label: &str, val: f64, unit: &str) {
        use std::fmt::Write;
        if unit.is_empty() {
            let _ = writeln!(self.buf, "  {:<29} {:>12.2}", label, val);
        } else {
            let _ = writeln!(self.buf, "  {:<29} {:>12.2} {}", label, val, unit);
        }
    }

    fn field_bytes(&mut self, label: &str, bytes: u64) {
        use std::fmt::Write;
        let _ = writeln!(
            self.buf,
            "  {:<29} {:>12} bytes (~{:.2} KB)",
            label,
            bytes,
            bytes_to_kb(bytes)
        );
    }

    fn custom_row(&mut self, row: &str) {
        self.buf.push_str(row);
        self.buf.push('\n');
    }

    fn finish(mut self) -> String {
        if self.buf.ends_with('\n') {
            self.buf.pop();
        }
        self.buf
    }
}

#[cfg(feature = "jit-stats")]
impl JitStats {
    pub fn record_compile(
        &mut self,
        guest_insts: u64,
        guest_bytes: u64,
        host_bytes: usize,
        stop_reason: BlockStopReason,
    ) {
        self.compiled_blocks += 1;
        self.compiled_guest_insts += guest_insts;
        self.compiled_guest_bytes += guest_bytes;
        self.host_code_bytes += host_bytes as u64;
        self.record_stop_reason(stop_reason);
    }

    pub fn record_unsupported_first_instr(&mut self, stop_reason: BlockStopReason) {
        self.record_stop_reason(stop_reason);
    }

    fn record_stop_reason(&mut self, stop_reason: BlockStopReason) {
        match stop_reason {
            BlockStopReason::DirectJump(_) => self.direct_jumps += 1,
            BlockStopReason::Branch(_) => self.branches += 1,
            BlockStopReason::DynamicJump(_) => self.dynamic_jumps += 1,
            BlockStopReason::BlockLimit => self.block_limits += 1,
            BlockStopReason::DecodeFault => self.decode_faults += 1,
            BlockStopReason::Unsupported(instr) => {
                self.total_unsupported += 1;
                *self.unsupported_insts.entry(instr).or_insert(0) += 1;
            }
        }
    }

    #[inline]
    pub fn record_jit_cycles(&mut self, cycles: u64) {
        self.jit_cycles += cycles;
    }

    #[inline]
    pub fn record_interpreter_cycles(&mut self, cycles: u64) {
        self.interpreter_cycles += cycles;
    }

    #[inline]
    pub fn record_jit_invocation(&mut self) {
        self.jit_block_invocations += 1;
    }

    #[inline]
    pub fn record_out_of_budget(&mut self) {
        self.out_of_budget_exits += 1;
    }

    #[inline]
    pub fn record_exception_exit(&mut self) {
        self.exception_exits += 1;
    }

    #[inline]
    pub fn record_chain_immediate(&mut self) {
        self.chained_immediate += 1;
    }

    #[inline]
    pub fn record_chain_forward_queued(&mut self) {
        self.unresolved_chains_queued += 1;
    }

    #[inline]
    pub fn record_chain_forward_resolved(&mut self, count: usize) {
        self.unresolved_chains_resolved += count as u64;
    }

    #[inline]
    pub fn record_epoch_invalidation(&mut self) {
        self.epoch_invalidations += 1;
    }

    pub fn total_cycles(&self) -> u64 {
        self.jit_cycles + self.interpreter_cycles
    }

    pub fn has_activity(&self) -> bool {
        self.total_cycles() > 0 || self.compiled_blocks > 0
    }

    pub fn format_report(
        &self,
        cache_hits: u64,
        cache_accesses: u64,
        buffer_allocated: usize,
        buffer_used: usize,
        buffer_peak: usize,
    ) -> String {
        let mut w = ReportWriter::new();
        let total_cycles = self.total_cycles();
        let normal_exits = self
            .jit_block_invocations
            .saturating_sub(self.out_of_budget_exits + self.exception_exits);
        let total_terminations = self.direct_jumps
            + self.branches
            + self.dynamic_jumps
            + self.block_limits
            + self.decode_faults
            + self.total_unsupported;
        let total_patches = self.chained_immediate + self.unresolved_chains_queued;

        w.line("======================= JIT Performance Stats =======================");

        w.section("Execution & Coverage");
        w.field_u64("Total Guest Cycles:", total_cycles);
        w.field_pct(
            "- JIT Executed Cycles:",
            self.jit_cycles,
            safe_pct(self.jit_cycles, total_cycles),
        );
        w.field_pct(
            "- Interpreter Fallback:",
            self.interpreter_cycles,
            safe_pct(self.interpreter_cycles, total_cycles),
        );
        w.field_u64("JIT Block Invocations:", self.jit_block_invocations);
        w.field_pct(
            "- Normal Block Exits:",
            normal_exits,
            safe_pct(normal_exits, self.jit_block_invocations),
        );
        w.field_pct(
            "- Budget Exhaustion Exits:",
            self.out_of_budget_exits,
            safe_pct(self.out_of_budget_exits, self.jit_block_invocations),
        );
        w.field_pct(
            "- Exception / Longjmp Exits:",
            self.exception_exits,
            safe_pct(self.exception_exits, self.jit_block_invocations),
        );

        w.section("Compilation Quality");
        w.field_u64("Compiled Basic Blocks:", self.compiled_blocks);
        w.field_u64("Compiled Guest Instructions:", self.compiled_guest_insts);
        w.field_bytes("Compiled Guest Code Size:", self.compiled_guest_bytes);
        w.field_f64(
            "Avg Instructions / Block:",
            safe_ratio(self.compiled_guest_insts, self.compiled_blocks),
            "",
        );
        w.field_f64(
            "Avg Guest Instruction Size:",
            safe_ratio(self.compiled_guest_bytes, self.compiled_guest_insts),
            "bytes (16-bit / 32-bit mix)",
        );
        w.field_bytes("Host Code Emitted:", self.host_code_bytes);
        w.field_f64(
            "Avg Host Bytes / Guest Inst:",
            safe_ratio(self.host_code_bytes, self.compiled_guest_insts),
            "bytes",
        );
        w.field_f64(
            "Code Expansion Ratio:",
            safe_ratio(self.host_code_bytes, self.compiled_guest_bytes),
            "x (Host bytes / Guest bytes)",
        );

        w.section("Block Termination / Stop Reasons");
        w.field_u64("Total Stop Observations:", total_terminations);
        w.field_pct(
            "- Direct Jumps (jal, c.j):",
            self.direct_jumps,
            safe_pct(self.direct_jumps, total_terminations),
        );
        w.field_pct(
            "- Branches (beq, bne...):",
            self.branches,
            safe_pct(self.branches, total_terminations),
        );
        w.field_pct(
            "- Dynamic Jumps (jalr, c.jr):",
            self.dynamic_jumps,
            safe_pct(self.dynamic_jumps, total_terminations),
        );
        w.field_pct(
            &format!("- Block Limit Reached ({}):", MAX_BLOCK_INSTRUCTIONS),
            self.block_limits,
            safe_pct(self.block_limits, total_terminations),
        );
        w.field_pct(
            "- Decode / Fetch Faults:",
            self.decode_faults,
            safe_pct(self.decode_faults, total_terminations),
        );
        w.field_pct(
            "- Unsupported Instructions:",
            self.total_unsupported,
            safe_pct(self.total_unsupported, total_terminations),
        );

        if !self.unsupported_insts.is_empty() {
            w.line("    -------------------------------------------------------");
            w.line("    * Top Unsupported Instructions (Priority for JIT impl):");
            let mut sorted: Vec<_> = self.unsupported_insts.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
            for (idx, (instr, count)) in sorted.iter().take(10).enumerate() {
                let name = instr.name();
                let pct = safe_pct(**count, self.total_unsupported);
                w.custom_row(&format!(
                    "      {:2}. {:<15} {:>8} ({:6.2}%)",
                    idx + 1,
                    name,
                    count,
                    pct
                ));
            }
            w.line("    -------------------------------------------------------");
        }

        w.section("Block Chaining & Cache");
        w.field_u64("Total Jump Patches Emitted:", total_patches);
        w.field_pct(
            "- Chained Immediately:",
            self.chained_immediate,
            safe_pct(self.chained_immediate, total_patches),
        );
        w.field_pct(
            "- Forward Chains Queued:",
            self.unresolved_chains_queued,
            safe_pct(self.unresolved_chains_queued, total_patches),
        );
        w.field_pct_suffix(
            "Forward Chains Resolved:",
            self.unresolved_chains_resolved,
            safe_pct(
                self.unresolved_chains_resolved,
                self.unresolved_chains_queued,
            ),
            "of queued",
        );
        w.custom_row(&format!(
            "  {:<29} {:>12.2}% (Hits: {}, Accesses: {})",
            "BB Cache Hit Rate:",
            safe_pct(cache_hits, cache_accesses),
            cache_hits,
            cache_accesses
        ));
        w.field_u64("iCache Epoch Invalidations:", self.epoch_invalidations);
        w.custom_row(&format!(
            "  {:<29} {:>9.2} MB / {:.2} MB (Peak: {:.2} MB)",
            "JIT Buffer Memory:",
            bytes_to_mb(buffer_used),
            bytes_to_mb(buffer_allocated),
            bytes_to_mb(buffer_peak)
        ));
        w.line("=====================================================================");

        w.finish()
    }
}

/// Zero-sized dummy implementation when feature `jit-stats` is disabled.
/// All methods are inlined as complete no-ops with zero runtime overhead.
#[cfg(not(feature = "jit-stats"))]
#[derive(Default, Debug, Clone, Copy)]
pub struct JitStats;

#[cfg(not(feature = "jit-stats"))]
impl JitStats {
    #[inline(always)]
    pub fn record_compile(
        &mut self,
        _guest_insts: u64,
        _guest_bytes: u64,
        _host_bytes: usize,
        _stop_reason: BlockStopReason,
    ) {
    }

    #[inline(always)]
    pub fn record_unsupported_first_instr(&mut self, _stop_reason: BlockStopReason) {}

    #[inline(always)]
    pub fn record_jit_cycles(&mut self, _cycles: u64) {}

    #[inline(always)]
    pub fn record_interpreter_cycles(&mut self, _cycles: u64) {}

    #[inline(always)]
    pub fn record_jit_invocation(&mut self) {}

    #[inline(always)]
    pub fn record_out_of_budget(&mut self) {}

    #[inline(always)]
    pub fn record_exception_exit(&mut self) {}

    #[inline(always)]
    pub fn record_chain_immediate(&mut self) {}

    #[inline(always)]
    pub fn record_chain_forward_queued(&mut self) {}

    #[inline(always)]
    pub fn record_chain_forward_resolved(&mut self, _count: usize) {}

    #[inline(always)]
    pub fn record_epoch_invalidation(&mut self) {}

    #[inline(always)]
    pub fn has_activity(&self) -> bool {
        false
    }

    #[inline(always)]
    pub fn format_report(
        &self,
        _cache_hits: u64,
        _cache_accesses: u64,
        _buffer_allocated: usize,
        _buffer_used: usize,
        _buffer_peak: usize,
    ) -> String {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_math_helpers() {
        assert_eq!(safe_ratio(10, 0), 0.0);
        assert_eq!(safe_ratio(10, 2), 5.0);
        assert_eq!(safe_pct(1, 0), 0.0);
        assert_eq!(safe_pct(1, 4), 25.0);
        assert_eq!(bytes_to_kb(2048), 2.0);
        assert_eq!(bytes_to_mb(2 * 1024 * 1024), 2.0);
    }

    #[test]
    fn test_stats_recording_and_formatting() {
        let mut stats = JitStats::default();
        stats.record_compile(10, 36, 40, BlockStopReason::DirectJump(RiscvInstr::JAL));
        stats.record_compile(5, 14, 24, BlockStopReason::Branch(RiscvInstr::BEQ));
        stats.record_compile(3, 10, 16, BlockStopReason::Unsupported(RiscvInstr::CSRRW));
        stats.record_unsupported_first_instr(BlockStopReason::Unsupported(RiscvInstr::CSRRW));
        stats.record_unsupported_first_instr(BlockStopReason::Unsupported(RiscvInstr::ECALL));

        stats.record_jit_cycles(150);
        stats.record_interpreter_cycles(50);
        stats.record_jit_invocation();
        stats.record_chain_immediate();
        stats.record_chain_forward_queued();
        stats.record_chain_forward_resolved(1);
        stats.record_epoch_invalidation();

        #[cfg(feature = "jit-stats")]
        {
            assert_eq!(stats.compiled_blocks, 3);
            assert_eq!(stats.compiled_guest_insts, 18);
            assert_eq!(stats.compiled_guest_bytes, 60);
            assert_eq!(stats.host_code_bytes, 80);
            assert_eq!(stats.total_cycles(), 200);
            assert_eq!(stats.total_unsupported, 3);
            assert_eq!(stats.unsupported_insts[&RiscvInstr::CSRRW], 2);
            assert_eq!(stats.unsupported_insts[&RiscvInstr::ECALL], 1);

            let report = stats.format_report(90, 100, 2 * 1024 * 1024, 80, 80);
            assert!(report.contains("JIT Performance Stats"));
            assert!(report.contains("Normal Block Exits"));
            assert!(report.contains("Budget Exhaustion Exits"));
            assert!(report.contains("csrrw"));
            assert!(report.contains("ecall"));
            assert!(report.contains("75.00%")); // JIT coverage
            assert!(report.contains("Code Expansion Ratio"));
        }

        #[cfg(not(feature = "jit-stats"))]
        {
            assert!(!stats.has_activity());
            assert_eq!(stats.format_report(0, 0, 0, 0, 0), "");
        }
    }
}
