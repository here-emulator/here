use std::fmt::Write;

use smallvec::SmallVec;

use here_types::WordType;

use super::TargetInst;

/// Guest control-flow edge a patched branch implements.
///
/// The label names the guest successor, not the final machine target: chain
/// patching replaces the local exit with another block's address at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsmEdge {
    OutOfBudget,
    Direct { target: WordType },
    Then { target: WordType },
    Else { target: WordType },
}

/// A value that only becomes known after the instruction was emitted.
///
/// Patches are recorded by byte offset, so they stay in sync with the
/// instruction stream without touching any rendered text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsmPatch {
    /// Raw immediate, e.g. the frame size of the stack allocation prologue.
    Imm(i64),
    /// Branch target as an offset into the current code buffer, plus the guest
    /// edge it represents.
    Target { offset: usize, edge: AsmEdge },
}

struct AsmEntry<I: TargetInst> {
    offset: usize,
    inst: I,
    patch: Option<AsmPatch>,
}

/// Structured host-assembly dump.
///
/// Instructions are recorded while code is emitted, patches are applied to the
/// matching entry, and the text is rendered once at the end. Text is never
/// produced before all patches are known.
pub struct AsmBuf<I: TargetInst> {
    entries: Vec<AsmEntry<I>>,
}

impl<I: TargetInst> Default for AsmBuf<I> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<I: TargetInst> AsmBuf<I> {
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn push(&mut self, offset: usize, inst: I) {
        debug_assert!(
            self.entries
                .last()
                .is_none_or(|entry| entry.offset < offset),
            "asm entries must be pushed in strictly increasing offset order"
        );
        self.entries.push(AsmEntry {
            offset,
            inst,
            patch: None,
        });
    }

    pub fn patch(&mut self, offset: usize, patch: AsmPatch) {
        let index = self
            .entries
            .binary_search_by_key(&offset, |entry| entry.offset)
            .unwrap_or_else(|_| panic!("asm patch at offset {offset:#x} has no instruction"));
        self.entries[index].patch = Some(patch);
    }

    pub fn render(&self, out: &mut String, indent: &str) {
        let mut labels: SmallVec<[(usize, AsmEdge); 4]> = self
            .entries
            .iter()
            .filter_map(|entry| match entry.patch {
                Some(AsmPatch::Target { offset, edge }) => Some((offset, edge)),
                _ => None,
            })
            .collect();
        labels.sort_by_key(|&(offset, _)| offset);
        labels.dedup();

        let mut next_label = 0;
        for entry in &self.entries {
            while next_label < labels.len() && labels[next_label].0 == entry.offset {
                write_edge_label(out, labels[next_label].1);
                out.push_str(":\n");
                next_label += 1;
            }

            out.push_str(indent);
            entry.inst.render_asm(out, entry.patch);
            out.push('\n');
        }

        debug_assert_eq!(
            next_label,
            labels.len(),
            "asm label target does not match an emitted instruction"
        );
    }
}

/// Writes the semantic label for a guest edge, without the trailing colon.
pub fn write_edge_label(out: &mut String, edge: AsmEdge) {
    let _ = match edge {
        AsmEdge::OutOfBudget => out.write_str(".L_oob"),
        AsmEdge::Direct { target } => write!(out, ".L_direct_{target:x}"),
        AsmEdge::Then { target } => write!(out, ".L_then_{target:x}"),
        AsmEdge::Else { target } => write!(out, ".L_else_{target:x}"),
    };
}
