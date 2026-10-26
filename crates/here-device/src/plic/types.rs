use here_types::WordType;

pub type PeriphIrqId = u32;

pub(super) const PLIC_SPEC_MAX_INTERRUPT_SOURCES: usize = 1024;
pub(super) const VIRT_MAX_INTERRUPTS: usize = 64;
pub(super) const PLIC_SPEC_MAX_CONTEXTS: usize = 15872;
pub(super) const VIRT_MAX_CONTEXTS: usize = 16;

const _: () = assert!(VIRT_MAX_INTERRUPTS <= PLIC_SPEC_MAX_INTERRUPT_SOURCES);
const _: () = assert!(VIRT_MAX_CONTEXTS <= PLIC_SPEC_MAX_CONTEXTS);

pub(super) type PlicPriority = u32;
pub(super) type PlicRegisterWord = u32;
pub type PlicContextId = usize;
pub(super) type PlicRegisterIndex = usize;

pub(super) const REGISTER_BYTES: WordType = size_of::<PlicRegisterWord>() as WordType;
pub(super) const INTERRUPTS_PER_REGISTER: usize = PlicRegisterWord::BITS as usize;
pub(super) const INTERRUPT_SOURCE_ZERO: PeriphIrqId = 0;
pub(super) const NO_PENDING_INTERRUPT: PeriphIrqId = 0;
pub(super) const PLIC_INTERRUPT_WORDS: usize = VIRT_MAX_INTERRUPTS / INTERRUPTS_PER_REGISTER;
pub(super) const CLAIM_COMPLETE_REGISTER_INDEX: PlicRegisterIndex = 1;

/// Per-context state: enable bits plus the notification threshold.
///
/// Claim/complete state is modeled by `PLICLayout::interrupt_sources_busy`,
/// because an interrupt source may be claimed by only one context at a time.
pub(super) struct PLICContext {
    pub(super) enable: PLICBitReg,
    pub(super) priority_threshold: PlicPriority,
}

impl PLICContext {
    pub(super) fn new() -> Self {
        Self {
            enable: PLICBitReg::new(),
            priority_threshold: 0,
        }
    }
}

/// A compact array-backed PLIC bit register.
///
/// The pending register and each context's enable register share the same
/// word layout, so all bit-level operations live here.
pub(super) struct PLICBitReg {
    bits: [PlicRegisterWord; PLIC_INTERRUPT_WORDS],
}

impl PLICBitReg {
    pub(super) fn new() -> Self {
        Self {
            bits: [PlicRegisterWord::default(); PLIC_INTERRUPT_WORDS],
        }
    }

    #[inline]
    pub(super) fn set_bit(&mut self, interrupt_id: PeriphIrqId) {
        let (index, mask) = interrupt_word_and_mask(interrupt_id);
        self.bits[index] |= mask;
    }

    #[inline]
    pub(super) fn clear_bit(&mut self, interrupt_id: PeriphIrqId) {
        let (index, mask) = interrupt_word_and_mask(interrupt_id);
        self.bits[index] &= !mask;
    }

    #[inline]
    pub(super) fn get_bit(&self, interrupt_id: PeriphIrqId) -> bool {
        let (index, mask) = interrupt_word_and_mask(interrupt_id);
        (self.bits[index] & mask) != 0
    }

    #[inline]
    pub(super) fn take_bit(&mut self, interrupt_id: PeriphIrqId) -> bool {
        let (index, mask) = interrupt_word_and_mask(interrupt_id);
        let old = self.bits[index];
        self.bits[index] &= !mask;
        old & mask != 0
    }

    #[inline]
    pub(super) fn read_word(&self, index: PlicRegisterIndex) -> PlicRegisterWord {
        self.bits[index]
    }

    #[inline]
    pub(super) fn write_word(&mut self, index: PlicRegisterIndex, value: PlicRegisterWord) {
        self.bits[index] = value;
    }
}

/// Decoded PLIC register address.
///
/// `PLIC::read_impl` and `PLIC::write_impl` decode an MMIO offset into this
/// enum, then delegate behavior to `PLICLayout`.
#[derive(Clone, Copy)]
pub(super) enum PlicRegister {
    Priority(PeriphIrqId),
    Pending(PlicRegisterIndex),
    Enable {
        context_id: PlicContextId,
        word_index: PlicRegisterIndex,
    },
    Threshold(PlicContextId),
    ClaimComplete(PlicContextId),
}

pub(super) fn source_zero_word_index() -> PlicRegisterIndex {
    let (word_index, _) = interrupt_word_and_mask(INTERRUPT_SOURCE_ZERO);
    word_index
}

pub(super) fn source_zero_bit_mask() -> PlicRegisterWord {
    let (_, mask) = interrupt_word_and_mask(INTERRUPT_SOURCE_ZERO);
    mask
}

fn interrupt_word_and_mask(interrupt_id: PeriphIrqId) -> (PlicRegisterIndex, PlicRegisterWord) {
    let interrupt_id = interrupt_id as usize;
    let word_index = interrupt_id / INTERRUPTS_PER_REGISTER;
    let bit_index = interrupt_id % INTERRUPTS_PER_REGISTER;
    (word_index, 1u32 << bit_index)
}
