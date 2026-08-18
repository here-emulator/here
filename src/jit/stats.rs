#[cfg(feature = "jit-stats")]
use std::collections::BTreeMap;

use crate::isa::riscv::trap::Exception;

use super::engine::TranslationStop;

#[derive(Default)]
pub(super) struct Stats {
    executed_blocks: u64,
    retired_instrs: u64,
    detailed: DetailedStats,
}

impl Stats {
    #[inline]
    pub(super) fn record_execution(&mut self, retired_instrs: u64) {
        self.executed_blocks = self.executed_blocks.wrapping_add(1);
        self.retired_instrs = self.retired_instrs.wrapping_add(retired_instrs);
    }

    #[inline]
    pub(super) fn record_translation(&mut self, translated_instrs: u64, stop: TranslationStop) {
        self.detailed.record_translation(translated_instrs, stop);
    }

    #[inline]
    pub(super) fn record_compiled_cache_hit(&mut self) {
        self.detailed.record_compiled_cache_hit();
    }

    #[inline]
    pub(super) fn record_exception_exit(&mut self, retired_instrs: u64, exception: Exception) {
        self.detailed
            .record_exception_exit(retired_instrs, exception);
    }

    pub(super) fn log(&self, bb_cache_hits: u64, bb_cache_accesses: u64) {
        log::info!("JIT statistics:");
        log::info!("  execution:");
        log::info!("    blocks executed: {}", self.executed_blocks);
        log::info!("    instructions retired: {}", self.retired_instrs);
        log::info!(
            "    average retired instructions per block: {:.3}",
            ratio(self.retired_instrs, self.executed_blocks)
        );
        log::info!("  cache:");
        log::info!(
            "    lookups: {bb_cache_accesses}, hits: {bb_cache_hits} ({:.3}%)",
            percentage(bb_cache_hits, bb_cache_accesses)
        );

        self.detailed
            .log(self.executed_blocks, self.retired_instrs, bb_cache_accesses);
    }
}

#[cfg(feature = "jit-stats")]
#[derive(Default)]
struct DetailedStats {
    unsupported_instrs: BTreeMap<crate::isa::riscv::instruction::instr_table::RiscvInstr, u64>,
    fetch_decode_failures: u64,
    instruction_limit_blocks: u64,
    control_flow_blocks: u64,
    translated_blocks: u64,
    translated_instrs: u64,
    compiled_cache_hits: u64,
    exception_exits: u64,
    exception_retired_instrs: u64,
    exception_causes: BTreeMap<Exception, u64>,
}

#[cfg(feature = "jit-stats")]
impl DetailedStats {
    #[inline]
    fn record_translation(&mut self, translated_instrs: u64, stop: TranslationStop) {
        debug_assert_ne!(translated_instrs, 0);

        self.translated_blocks = self.translated_blocks.wrapping_add(1);
        self.translated_instrs = self.translated_instrs.wrapping_add(translated_instrs);

        match stop {
            TranslationStop::DecodeFailure => {
                self.fetch_decode_failures = self.fetch_decode_failures.wrapping_add(1);
            }
            TranslationStop::UnsupportedInstruction(instr) => {
                *self.unsupported_instrs.entry(instr).or_default() += 1;
            }
            TranslationStop::InstructionLimit => {
                self.instruction_limit_blocks = self.instruction_limit_blocks.wrapping_add(1);
            }
            TranslationStop::ControlFlow => {
                self.control_flow_blocks = self.control_flow_blocks.wrapping_add(1);
            }
        }
    }

    #[inline]
    fn record_compiled_cache_hit(&mut self) {
        self.compiled_cache_hits = self.compiled_cache_hits.wrapping_add(1);
    }

    #[inline]
    fn record_exception_exit(&mut self, retired_instrs: u64, exception: Exception) {
        self.exception_exits = self.exception_exits.wrapping_add(1);
        self.exception_retired_instrs = self.exception_retired_instrs.wrapping_add(retired_instrs);
        *self.exception_causes.entry(exception).or_default() += 1;
    }

    fn log(&self, executed_blocks: u64, retired_instrs: u64, bb_cache_accesses: u64) {
        let attempted_instrs = retired_instrs.wrapping_add(self.exception_exits);
        log::info!(
            "    instructions attempted: {attempted_instrs} (retirement rate {:.3}%)",
            percentage(retired_instrs, attempted_instrs)
        );
        let compiled_cache_hit_rate = percentage(self.compiled_cache_hits, bb_cache_accesses);
        log::info!(
            "    compiled block hits: {}/{} ({compiled_cache_hit_rate:.3}%)",
            self.compiled_cache_hits,
            bb_cache_accesses
        );
        log::info!(
            "    exception exits: {} ({:.3}% of blocks)",
            self.exception_exits,
            percentage(self.exception_exits, executed_blocks)
        );
        log::info!(
            "    average retired instructions before exception: {:.2}",
            ratio(self.exception_retired_instrs, self.exception_exits)
        );

        let mut exceptions: Vec<_> = self.exception_causes.iter().collect();
        exceptions.sort_unstable_by(|(name_a, count_a), (name_b, count_b)| {
            count_b.cmp(count_a).then_with(|| name_a.cmp(name_b))
        });
        if !exceptions.is_empty() {
            log::info!("  exception causes:");
            for (exception, count) in exceptions {
                log::info!(
                    "    {exception:<32} {count:>8} ({:>5.1}%)",
                    percentage(*count, self.exception_exits)
                );
            }
        }

        log::info!("  translation:");
        log::info!("    blocks translated: {}", self.translated_blocks);
        log::info!(
            "    instructions translated: {} (average {:.2} per block)",
            self.translated_instrs,
            ratio(self.translated_instrs, self.translated_blocks)
        );

        let stop_count = self.translated_blocks;
        let mut instrs: Vec<_> = self.unsupported_instrs.iter().collect();
        instrs.sort_unstable_by(|(name_a, count_a), (name_b, count_b)| {
            count_b.cmp(count_a).then_with(|| name_a.cmp(name_b))
        });

        log::info!("    translation stop reasons: {stop_count}");
        log::info!(
            "      fetch/decode failure: {} ({:.1}%)",
            self.fetch_decode_failures,
            percentage(self.fetch_decode_failures, stop_count)
        );
        log::info!(
            "      instruction limit: {} ({:.1}%)",
            self.instruction_limit_blocks,
            percentage(self.instruction_limit_blocks, stop_count)
        );
        log::info!(
            "      control flow: {} ({:.1}%)",
            self.control_flow_blocks,
            percentage(self.control_flow_blocks, stop_count)
        );

        if !instrs.is_empty() {
            log::info!("      unsupported instructions:");
            for (instr, count) in instrs {
                log::info!(
                    "        {:<16} {count:>8} ({:>5.1}%)",
                    instr.name(),
                    percentage(*count, stop_count)
                );
            }
        }
    }
}

#[cfg(not(feature = "jit-stats"))]
#[derive(Default)]
struct DetailedStats;

#[cfg(not(feature = "jit-stats"))]
impl DetailedStats {
    #[inline(always)]
    fn record_translation(&mut self, _translated_instrs: u64, _stop: TranslationStop) {}

    #[inline(always)]
    fn record_compiled_cache_hit(&mut self) {}

    #[inline(always)]
    fn record_exception_exit(&mut self, _retired_instrs: u64, _exception: Exception) {}

    #[inline(always)]
    fn log(&self, _executed_blocks: u64, _retired_instrs: u64, _bb_cache_accesses: u64) {}
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn percentage(numerator: u64, denominator: u64) -> f64 {
    ratio(numerator, denominator) * 100.0
}

#[cfg(all(test, feature = "jit-stats"))]
mod tests {
    use super::*;
    use crate::isa::riscv::trap::Exception;

    #[test]
    fn records_translation_and_exception_aggregates() {
        let mut stats = Stats::default();

        stats.record_translation(4, TranslationStop::InstructionLimit);
        stats.record_translation(2, TranslationStop::DecodeFailure);
        stats.record_translation(1, TranslationStop::ControlFlow);
        stats.record_exception_exit(2, Exception::LoadFault);
        stats.record_exception_exit(0, Exception::LoadFault);

        assert_eq!(stats.detailed.translated_blocks, 3);
        assert_eq!(stats.detailed.translated_instrs, 7);
        assert_eq!(stats.detailed.instruction_limit_blocks, 1);
        assert_eq!(stats.detailed.fetch_decode_failures, 1);
        assert_eq!(stats.detailed.control_flow_blocks, 1);
        assert_eq!(stats.detailed.exception_exits, 2);
        assert_eq!(stats.detailed.exception_retired_instrs, 2);
        assert_eq!(
            stats.detailed.exception_causes.get(&Exception::LoadFault),
            Some(&2)
        );
        assert_eq!(ratio(6, 2), 3.0);
        assert_eq!(percentage(1, 4), 25.0);
        assert_eq!(ratio(1, 0), 0.0);
    }
}
