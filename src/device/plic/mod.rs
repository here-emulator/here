pub mod types;

use std::{hint::unlikely, ptr::NonNull};

use crate::{
    board::virt::RiscvIRQSource,
    config::arch_config::WordType,
    device::{DeviceTrait, MemError, PlicDevice},
};
use bit_set::BitSet;

#[cfg(test)]
use types::INTERRUPTS_PER_REGISTER;
pub use types::PeriphIrqId;
use types::{
    CLAIM_COMPLETE_REGISTER_INDEX, INTERRUPT_SOURCE_ZERO, NO_PENDING_INTERRUPT,
    PLIC_INTERRUPT_WORDS, PLICBitReg, PLICContext, PlicContextId, PlicPriority, PlicRegister,
    PlicRegisterIndex, PlicRegisterWord, REGISTER_BYTES, VIRT_MAX_CONTEXTS, VIRT_MAX_INTERRUPTS,
    source_zero_bit_mask, source_zero_word_index,
};

const PRIORITY_OFFSET: WordType = 0;
const PENDING_BIT_OFFSET: WordType = 0x001000;
const CONTEXT_ENABLE_BIT_OFFSET: WordType = 0x002000;
const CONTEXT_ENABLE_BIT_SIZE: WordType = 0x80;
const CONTEXT_CONFIG_OFFSET: WordType = 0x200000;
const CONTEXT_CONFIG_SIZE: WordType = 0x1000;

/// MMIO region size for the PLIC device.
pub const PLIC_SIZE: WordType = 0x400_0000;

/*
    - priority (0x000000 - 0x000ffc)
        base + 0x000000: Reserved (interrupt source 0 does not exist)
        base + 0x000004: Interrupt source 1 priority
        base + 0x000008: Interrupt source 2 priority
        ...
        base + 0x000FFC: Interrupt source 1023 priority

    - pending (0x001000 - 0x00107c)
        base + 0x001000: Interrupt Pending bit 0-31
        base + 0x00107C: Interrupt Pending bit 992-1023

    - enable (0x002000 - 0x1FFFFC)
        - Context 0
        base + 0x002000: Enable bits for sources 0-31 on context 0
        base + 0x002004: Enable bits for sources 32-63 on context 0
        ...
        base + 0x00207C: Enable bits for sources 992-1023 on context 0

        - Context 2
        ...

        - Context 15871
        base + 0x1F1F80: Enable bits for sources 0-31 on context 15871
        base + 0x1F1F84: Enable bits for sources 32-63 on context 15871
        ...
        base + 0x1F1FFC: Enable bits for sources 992-1023 on context 15871

    - claim
        - Context 0
        base + 0x200000: Priority threshold for context 0
        base + 0x200004: Claim/complete for context 0

        - Context 1
        base + 0x201000: Priority threshold for context 1
        base + 0x201004: Claim/complete for context 1

        - Context 2
        ...

        - context 15871
        base + 0x3FFF000: Priority threshold for context 15871
        base + 0x3FFF004: Claim/complete for context 15871
        ...
        base + 0x3FFFFFC: Reserved
*/
/// Register-level PLIC state and arbitration logic.
///
/// `PLIC` owns CPU interrupt lines and MMIO plumbing; this layout object owns
/// the actual PLIC registers plus claim/complete arbitration state.
pub struct PLICLayout {
    priority: [PlicPriority; VIRT_MAX_INTERRUPTS],
    pending: PLICBitReg,
    /// Latest electrical level observed at each interrupt gateway.
    ///
    /// This is separate from `pending`: a low level cannot retract an accepted
    /// request, while a level that remains high must create another request
    /// after the previous one is claimed and completed.
    source_level: PLICBitReg,
    contexts: [PLICContext; VIRT_MAX_CONTEXTS],
    /// Interrupt source ids sorted by descending priority, then ascending source id.
    priority_order: [PeriphIrqId; VIRT_MAX_INTERRUPTS],
    /// Sources claimed by a context but not completed yet.
    interrupt_sources_busy: BitSet,
}

impl PLICLayout {
    pub fn new() -> Self {
        let priority = [0; VIRT_MAX_INTERRUPTS];
        let priority_order = core::array::from_fn(|interrupt_id| interrupt_id as PeriphIrqId);

        Self {
            priority,
            pending: PLICBitReg::new(),
            source_level: PLICBitReg::new(),
            contexts: core::array::from_fn(|_| PLICContext::new()),
            priority_order,
            interrupt_sources_busy: BitSet::with_capacity(VIRT_MAX_INTERRUPTS),
        }
    }

    #[inline]
    fn read_priority(&self, interrupt_id: PeriphIrqId) -> PlicPriority {
        self.priority[interrupt_id as usize]
    }

    #[inline]
    fn write_priority(&mut self, interrupt_id: PeriphIrqId, value: PlicPriority) {
        if value != self.priority[interrupt_id as usize] {
            self.priority[interrupt_id as usize] = value;
            self.sort_priority_order();
        }
    }

    fn sort_priority_order(&mut self) {
        self.priority_order.sort_by_key(|interrupt_id| {
            (
                core::cmp::Reverse(self.priority[*interrupt_id as usize]),
                *interrupt_id,
            )
        });
    }

    #[inline]
    fn read_pending_word(&self, word_index: PlicRegisterIndex) -> PlicRegisterWord {
        self.pending.read_word(word_index)
    }

    /// Update the electrical level observed at one interrupt gateway.
    ///
    /// A rising/high level creates at most one request while the source is not
    /// already pending or claimed. Dropping the line only changes the sampled
    /// level; an already accepted pending request remains pending until claim.
    fn set_source_level(&mut self, interrupt_id: PeriphIrqId, level: bool) {
        if level {
            self.source_level.set_bit(interrupt_id);
            if !self.pending.get_bit(interrupt_id)
                && !self.interrupt_sources_busy.contains(interrupt_id as usize)
            {
                self.pending.set_bit(interrupt_id);
            }
        } else {
            self.source_level.clear_bit(interrupt_id);
        }
    }

    #[inline]
    fn read_enable_word(
        &self,
        context_id: PlicContextId,
        word_index: PlicRegisterIndex,
    ) -> PlicRegisterWord {
        self.contexts[context_id].enable.read_word(word_index)
    }

    #[inline]
    fn write_enable_word(
        &mut self,
        context_id: PlicContextId,
        word_index: PlicRegisterIndex,
        value: PlicRegisterWord,
    ) {
        let value = if word_index == source_zero_word_index() {
            value & !source_zero_bit_mask()
        } else {
            value
        };
        self.contexts[context_id]
            .enable
            .write_word(word_index, value);
    }

    #[inline]
    fn is_enabled(&self, context_id: PlicContextId, interrupt_id: PeriphIrqId) -> bool {
        self.contexts[context_id].enable.get_bit(interrupt_id)
    }

    #[inline]
    fn read_priority_threshold(&self, context_id: PlicContextId) -> PlicPriority {
        self.contexts[context_id].priority_threshold
    }

    #[inline]
    fn write_priority_threshold(&mut self, context_id: PlicContextId, value: PlicPriority) {
        self.contexts[context_id].priority_threshold = value;
    }

    /// Claim the highest-priority pending source enabled for this context.
    ///
    /// Per the PLIC spec, claim ignores the context threshold. A successful
    /// claim atomically clears the source pending bit and marks the source busy
    /// until completion.
    fn read_claim(&mut self, context_id: PlicContextId) -> PeriphIrqId {
        let Some(interrupt_id) = self.find_pending_interrupt(context_id, false) else {
            return NO_PENDING_INTERRUPT;
        };
        self.pending.take_bit(interrupt_id);
        self.interrupt_sources_busy.insert(interrupt_id as usize);
        interrupt_id
    }

    /// Complete a previously claimed source.
    ///
    /// Completion is ignored when the source id is invalid or disabled for the
    /// completing context.
    fn write_complete(&mut self, context_id: PlicContextId, interrupt_id: PeriphIrqId) {
        if !is_valid_interrupt_id(interrupt_id) || !self.is_enabled(context_id, interrupt_id) {
            return;
        }
        if self.interrupt_sources_busy.remove(interrupt_id as usize)
            && self.source_level.get_bit(interrupt_id)
        {
            self.pending.set_bit(interrupt_id);
        }
    }

    fn pending_interrupt_to_notify(&self, context_id: PlicContextId) -> Option<PeriphIrqId> {
        self.find_pending_interrupt(context_id, true)
    }

    /// Find the next eligible interrupt source in priority order.
    ///
    /// When `apply_threshold` is true this models notification eligibility.
    /// When false this models claim-register reads, where the threshold is not
    /// considered.
    fn find_pending_interrupt(
        &self,
        context_id: PlicContextId,
        apply_threshold: bool,
    ) -> Option<PeriphIrqId> {
        let threshold = self.contexts[context_id].priority_threshold;

        for interrupt_id in self.priority_order.iter().copied() {
            if interrupt_id == INTERRUPT_SOURCE_ZERO {
                continue;
            }

            let priority = self.priority[interrupt_id as usize];
            if priority == 0 {
                break;
            }

            if apply_threshold && priority <= threshold {
                continue;
            }

            if self.interrupt_sources_busy.contains(interrupt_id as usize) {
                continue;
            }

            if self.is_enabled(context_id, interrupt_id) && self.pending.get_bit(interrupt_id) {
                return Some(interrupt_id);
            }
        }

        None
    }

    /// Decode a 32-bit aligned PLIC MMIO offset into a typed register.
    fn decode_register(inner_addr: WordType) -> Option<PlicRegister> {
        if !is_register_aligned(inner_addr) {
            return None;
        }

        match inner_addr {
            PRIORITY_OFFSET..PENDING_BIT_OFFSET => {
                let interrupt_id = (inner_addr / REGISTER_BYTES) as PeriphIrqId;
                is_valid_interrupt_id(interrupt_id).then_some(PlicRegister::Priority(interrupt_id))
            }
            PENDING_BIT_OFFSET..CONTEXT_ENABLE_BIT_OFFSET => {
                let offset = inner_addr - PENDING_BIT_OFFSET;
                let word_index = (offset / REGISTER_BYTES) as usize;
                (word_index < PLIC_INTERRUPT_WORDS).then_some(PlicRegister::Pending(word_index))
            }
            CONTEXT_ENABLE_BIT_OFFSET..CONTEXT_CONFIG_OFFSET => {
                let context_id =
                    ((inner_addr - CONTEXT_ENABLE_BIT_OFFSET) / CONTEXT_ENABLE_BIT_SIZE) as usize;
                let offset = (inner_addr - CONTEXT_ENABLE_BIT_OFFSET) % CONTEXT_ENABLE_BIT_SIZE;
                let word_index = (offset / REGISTER_BYTES) as usize;

                if context_id < VIRT_MAX_CONTEXTS && word_index < PLIC_INTERRUPT_WORDS {
                    Some(PlicRegister::Enable {
                        context_id,
                        word_index,
                    })
                } else {
                    None
                }
            }
            CONTEXT_CONFIG_OFFSET..PLIC_SIZE => {
                let context_id =
                    ((inner_addr - CONTEXT_CONFIG_OFFSET) / CONTEXT_CONFIG_SIZE) as usize;
                let offset = (inner_addr - CONTEXT_CONFIG_OFFSET) % CONTEXT_CONFIG_SIZE;
                let register_index = (offset / REGISTER_BYTES) as usize;

                if context_id >= VIRT_MAX_CONTEXTS {
                    return None;
                }

                match register_index {
                    0 => Some(PlicRegister::Threshold(context_id)),
                    CLAIM_COMPLETE_REGISTER_INDEX => Some(PlicRegister::ClaimComplete(context_id)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn read_register(&mut self, register: PlicRegister) -> PlicRegisterWord {
        match register {
            PlicRegister::Priority(interrupt_id) => self.read_priority(interrupt_id),
            PlicRegister::Pending(word_index) => self.read_pending_word(word_index),
            PlicRegister::Enable {
                context_id,
                word_index,
            } => self.read_enable_word(context_id, word_index),
            PlicRegister::Threshold(context_id) => self.read_priority_threshold(context_id),
            PlicRegister::ClaimComplete(context_id) => {
                let interrupt_id = self.read_claim(context_id);
                log::debug!(
                    "[PLIC] Claim read ctx={} => id={}",
                    context_id,
                    interrupt_id
                );
                interrupt_id
            }
        }
    }

    fn write_register(&mut self, register: PlicRegister, value: PlicRegisterWord) {
        match register {
            PlicRegister::Priority(interrupt_id) => self.write_priority(interrupt_id, value),
            PlicRegister::Enable {
                context_id,
                word_index,
            } => self.write_enable_word(context_id, word_index, value),
            PlicRegister::Threshold(context_id) => {
                self.write_priority_threshold(context_id, value);
            }
            PlicRegister::ClaimComplete(context_id) => {
                log::debug!("[PLIC] Complete write ctx={} id={}", context_id, value);
                self.write_complete(context_id, value);
            }
            PlicRegister::Pending(_) => {}
        }
    }
}

fn is_valid_interrupt_id(interrupt_id: PeriphIrqId) -> bool {
    interrupt_id != INTERRUPT_SOURCE_ZERO && (interrupt_id as usize) < VIRT_MAX_INTERRUPTS
}

fn is_valid_context_id(context_id: PlicContextId) -> bool {
    context_id < VIRT_MAX_CONTEXTS
}

fn is_register_aligned(inner_addr: WordType) -> bool {
    inner_addr % REGISTER_BYTES == 0
}

/// Platform-Level Interrupt Controller device.
///
/// The device receives external interrupt source changes from peripherals,
/// exposes the PLIC MMIO register map, and drives per-context CPU IRQ lines.
pub struct PLIC {
    layout: PLICLayout,
    riscv_irq_line: [Option<crate::board::virt::IRQLine>; VIRT_MAX_CONTEXTS],
    riscv_irq_level: [Option<bool>; VIRT_MAX_CONTEXTS],
    // Non-owning pointers to boxed devices held by the board's DeviceArena.
    peripheral_irq_devices: [Option<NonNull<dyn crate::device::PlicDevice>>; VIRT_MAX_INTERRUPTS],
}

impl PLIC {
    pub fn new() -> Self {
        Self {
            layout: PLICLayout::new(),
            riscv_irq_line: core::array::from_fn(|_| None),
            riscv_irq_level: [None; VIRT_MAX_CONTEXTS],
            peripheral_irq_devices: core::array::from_fn(|_| None),
        }
    }

    // MMIO accesses update only internal PLIC state. CPU IRQ outputs are
    // driven at the next board batch boundary; calling back into the CPU from
    // its own MMIO access would alias its active mutable borrow.
    fn read_impl<T>(&mut self, inner_addr: WordType) -> Result<T, super::MemError>
    where
        T: crate::utils::UnsignedInteger,
    {
        if size_of::<T>() as WordType != REGISTER_BYTES {
            return Err(MemError::LoadFault);
        }

        let register = PLICLayout::decode_register(inner_addr).ok_or(MemError::LoadFault)?;
        if matches!(register, PlicRegister::ClaimComplete(_)) {
            self.sync_peripheral_irq_levels();
        }
        let data = self.layout.read_register(register);

        Ok(T::truncate_from(data))
    }

    fn write_impl<T>(&mut self, inner_addr: WordType, data: T) -> Result<(), super::MemError>
    where
        T: crate::utils::UnsignedInteger,
    {
        if size_of::<T>() as WordType != REGISTER_BYTES {
            return Err(MemError::StoreFault);
        }

        let register = PLICLayout::decode_register(inner_addr).ok_or(MemError::StoreFault)?;
        if matches!(register, PlicRegister::Pending(_)) {
            return Err(MemError::StoreFault);
        }

        if matches!(register, PlicRegister::ClaimComplete(_)) {
            self.sync_peripheral_irq_levels();
        }

        self.layout.write_register(register, data.truncate_to());

        Ok(())
    }

    /// Set the current level of an external interrupt source.
    fn set_interrupt_level(&mut self, interrupt_id: PeriphIrqId, level: bool) {
        if unlikely(!is_valid_interrupt_id(interrupt_id)) {
            return;
        }
        self.layout.set_source_level(interrupt_id, level);
    }

    pub(crate) fn register_device(
        &mut self,
        device: NonNull<dyn PlicDevice>,
        interrupt_id: PeriphIrqId,
    ) {
        assert!(is_valid_interrupt_id(interrupt_id));
        self.peripheral_irq_devices[interrupt_id as usize] = Some(device);
    }

    fn sync_peripheral_irq_levels(&mut self) {
        for (interrupt_id, device) in self.peripheral_irq_devices.iter_mut().enumerate() {
            let Some(device) = device else {
                continue;
            };

            let level = unsafe { device.as_mut() }.irq_level();
            self.layout
                .set_source_level(interrupt_id as PeriphIrqId, level);
        }
    }

    /// Refresh the target CPU interrupt line for one PLIC context.
    pub fn update_context_irq_line(&mut self, context_id: PlicContextId) -> Option<PeriphIrqId> {
        if unlikely(!is_valid_context_id(context_id)) {
            return None;
        }
        self.sync_peripheral_irq_levels();
        self.refresh_context_irq_line(context_id)
    }

    /// Refresh several CPU contexts from one coherent peripheral snapshot.
    pub fn update_context_irq_lines(&mut self, context_ids: &[PlicContextId]) {
        self.sync_peripheral_irq_levels();
        for &context_id in context_ids {
            if is_valid_context_id(context_id) {
                self.refresh_context_irq_line(context_id);
            }
        }
    }

    fn refresh_context_irq_line(&mut self, context_id: PlicContextId) -> Option<PeriphIrqId> {
        let pending_interrupt = self.layout.pending_interrupt_to_notify(context_id);
        self.set_context_irq_line(context_id, pending_interrupt.is_some());

        if let Some(interrupt_id) = pending_interrupt {
            log::trace!(
                "[PLIC] assert IRQ line ctx={} id={}",
                context_id,
                interrupt_id
            );
        }

        pending_interrupt
    }

    fn set_context_irq_line(&mut self, context_id: PlicContextId, level: bool) {
        if self.riscv_irq_level[context_id] == Some(level) {
            return;
        }
        self.riscv_irq_level[context_id] = Some(level);

        if let Some(irq_line) = &mut self.riscv_irq_line[context_id] {
            irq_line.set_irq(level);
        }
    }
}

impl DeviceTrait for PLIC {
    dispatch_read_write! { read_impl, write_impl }

    fn sync(&mut self) {
        // nothing to do.
    }
}

// Send the external interrupt resulting from the arbitration to the CPU through the IRQLine.
impl RiscvIRQSource for PLIC {
    fn set_irq_line(&mut self, line: crate::board::virt::IRQLine, id: usize) {
        assert!(id < VIRT_MAX_CONTEXTS);
        // plic external interrupt source id will be write to plic.claim register.
        self.riscv_irq_line[id] = Some(line);
        self.riscv_irq_level[id] = None;
        self.refresh_context_irq_line(id);
    }
}

#[cfg(test)]
mod test {
    use std::cell::Cell;

    use super::*;
    use crate::device::{PlicDevice, device_manager::DeviceArenaBuilder};

    // all methods go through mmio interface.
    impl PLIC {
        fn get_priority(&mut self, interrupt_id: WordType) -> Result<u32, MemError> {
            self.read_impl(interrupt_id * REGISTER_BYTES)
        }

        fn set_priority(&mut self, interrupt_id: WordType, value: u32) -> Result<(), MemError> {
            self.write_impl(interrupt_id * REGISTER_BYTES, value)
        }

        fn get_pending_bit(&mut self, interrupt_id: WordType) -> Result<bool, MemError> {
            let addr = PENDING_BIT_OFFSET
                + (interrupt_id / INTERRUPTS_PER_REGISTER as WordType * REGISTER_BYTES);
            let word = self.read_impl::<u32>(addr)?;
            let bit = interrupt_id % INTERRUPTS_PER_REGISTER as WordType;
            Ok((word & (1 << bit)) != 0)
        }

        fn get_enable_word(
            &mut self,
            context_id: WordType,
            word_index: WordType,
        ) -> Result<u32, MemError> {
            let addr = CONTEXT_ENABLE_BIT_OFFSET
                + (context_id * CONTEXT_ENABLE_BIT_SIZE)
                + (word_index * REGISTER_BYTES);
            self.read_impl::<u32>(addr)
        }

        fn set_enable_word(
            &mut self,
            context_id: WordType,
            word_index: WordType,
            value: u32,
        ) -> Result<(), MemError> {
            let addr = CONTEXT_ENABLE_BIT_OFFSET
                + (context_id * CONTEXT_ENABLE_BIT_SIZE)
                + (word_index * REGISTER_BYTES);
            self.write_impl(addr, value)
        }

        fn get_priority_threshold(&mut self, context_id: WordType) -> Result<u32, MemError> {
            let addr = CONTEXT_CONFIG_OFFSET + (context_id * CONTEXT_CONFIG_SIZE);
            self.read_impl::<u32>(addr)
        }

        fn get_claim_complete(&mut self, context_id: WordType) -> Result<u32, MemError> {
            let addr = CONTEXT_CONFIG_OFFSET
                + (context_id * CONTEXT_CONFIG_SIZE)
                + CLAIM_COMPLETE_REGISTER_INDEX as WordType * REGISTER_BYTES;
            self.read_impl::<u32>(addr)
        }

        fn set_priority_threshold(
            &mut self,
            context_id: WordType,
            value: u32,
        ) -> Result<(), MemError> {
            let addr = CONTEXT_CONFIG_OFFSET + (context_id * CONTEXT_CONFIG_SIZE);
            self.write_impl(addr, value)
        }

        fn set_claim_complete(
            &mut self,
            context_id: WordType,
            interrupt_id: u32,
        ) -> Result<(), MemError> {
            let addr = CONTEXT_CONFIG_OFFSET
                + (context_id * CONTEXT_CONFIG_SIZE)
                + CLAIM_COMPLETE_REGISTER_INDEX as WordType * REGISTER_BYTES;
            self.write_impl(addr, interrupt_id)
        }
    }

    fn pulse_interrupt(plic: &mut PLIC, interrupt_id: PeriphIrqId) {
        plic.set_interrupt_level(interrupt_id, true);
        plic.set_interrupt_level(interrupt_id, false);
    }

    #[test]
    fn plic_layout_test() {
        let mut plic = PLIC::new();

        // =======================
        // ====== priority =======
        // =======================``
        assert!(plic.set_priority(0, 0).is_err()); // (interrupt source 0 does not exist)
        assert!(
            plic.get_priority((VIRT_MAX_INTERRUPTS * size_of::<u32>()) as WordType)
                .is_err()
        ); // over max priority index.
        plic.set_priority(1, 5u32).unwrap();
        assert_eq!(plic.get_priority(1).unwrap(), 5u32);

        // =======================
        // ======= pending =======
        // =======================
        assert!(
            plic.write_impl(
                PENDING_BIT_OFFSET + 1 * size_of::<u32>() as WordType,
                0x1234_5678u32
            )
            .is_err()
        ); // pending is read-only
        assert_eq!(plic.get_pending_bit(5).unwrap(), false);
        assert!(
            plic.get_pending_bit(VIRT_MAX_INTERRUPTS as WordType)
                .is_err()
        ); // over max pending index
        assert!(
            plic.read_impl::<u32>(PENDING_BIT_OFFSET + VIRT_MAX_INTERRUPTS as WordType / 8)
                .is_err()
        ); // over max pending index

        // =======================
        // ===== enable bits =====
        // =======================
        plic.set_enable_word(0, 0, 0xffff_ffffu32).unwrap();
        assert_eq!(plic.get_enable_word(0, 0).unwrap(), 0xffff_fffe);
        plic.set_enable_word(0, 1, 0xdead_beefu32).unwrap();
        assert_eq!(plic.get_enable_word(0, 1).unwrap(), 0xdead_beefu32); // assert the value read is consistent with the one written.
        assert!(
            plic.set_enable_word(VIRT_MAX_CONTEXTS as WordType, 0, 0x1234_5678u32)
                .is_err()
        ); // over max context index
        assert!(
            plic.set_enable_word(0, VIRT_MAX_INTERRUPTS as WordType / 32, 0x1234_5678u32)
                .is_err()
        ); // over max context index

        // =======================
        // === context config ====
        // =======================
        plic.set_priority_threshold(0, 3).unwrap();
        assert_eq!(plic.get_priority_threshold(0).unwrap(), 3);
        assert!(
            plic.set_priority_threshold(VIRT_MAX_CONTEXTS as WordType, 0)
                .is_err()
        ); // over max context index

        assert_eq!(plic.get_claim_complete(0).unwrap(), 0);
        plic.set_priority(2, 4).unwrap();
        plic.set_enable_word(0, 0, 1 << 2).unwrap();
        pulse_interrupt(&mut plic, 2);
        assert_eq!(plic.update_context_irq_line(0), Some(2));
        assert_eq!(plic.get_claim_complete(0).unwrap(), 2);
        plic.set_claim_complete(0, 2).unwrap();
        assert_eq!(plic.get_claim_complete(0).unwrap(), 0);
        assert!(
            plic.get_claim_complete(VIRT_MAX_CONTEXTS as WordType)
                .is_err()
        );
        assert!(
            plic.set_claim_complete(VIRT_MAX_CONTEXTS as WordType, 0)
                .is_err()
        );
    }

    #[test]
    fn interrupt_test() {
        let mut plic = PLIC::new();
        plic.set_priority(1, 5).unwrap();
        plic.set_priority(2, 7).unwrap();
        pulse_interrupt(&mut plic, 1);
        pulse_interrupt(&mut plic, 2);
        assert!(plic.update_context_irq_line(0).is_none());

        // context 0 <- interrupt 2 (received)
        plic.set_enable_word(0, 0, 1 << 2).unwrap();
        assert_eq!(plic.update_context_irq_line(0), Some(2));
        assert_eq!(plic.get_claim_complete(0).unwrap(), 2);
        assert!(plic.update_context_irq_line(0).is_none()); // interrupt 2 is claimed

        // context 1 <- interrupt 1 (received)
        plic.set_enable_word(1, 0, 0xffffffff).unwrap();
        assert_eq!(plic.update_context_irq_line(1), Some(1));
        assert_eq!(plic.get_claim_complete(1).unwrap(), 1);
        // context 1 <- interrupt 1 (completed)
        plic.set_claim_complete(1, 1).unwrap();

        // A second request cannot be accepted while source 2 is claimed.
        pulse_interrupt(&mut plic, 2);
        assert!(plic.update_context_irq_line(1).is_none()); // interrupt 2 is not completed.

        // context 0 <- interrupt 2 (completed)
        plic.set_claim_complete(0, 2).unwrap();

        pulse_interrupt(&mut plic, 2);
        // context 1 <- interrupt 2 (received)
        assert_eq!(plic.update_context_irq_line(1), Some(2));
        assert_eq!(plic.get_claim_complete(1).unwrap(), 2);

        // context 0 <- None
        assert!(plic.update_context_irq_line(0).is_none());
    }

    #[test]
    fn claim_ignores_threshold_but_notification_does_not() {
        let mut plic = PLIC::new();
        plic.set_priority(3, 4).unwrap();
        plic.set_enable_word(0, 0, 1 << 3).unwrap();
        plic.set_priority_threshold(0, 4).unwrap();
        pulse_interrupt(&mut plic, 3);

        assert!(plic.update_context_irq_line(0).is_none());
        assert_eq!(plic.get_claim_complete(0).unwrap(), 3);
    }

    #[test]
    fn lower_interrupt_id_wins_priority_ties() {
        let mut plic = PLIC::new();
        plic.set_priority(4, 7).unwrap();
        plic.set_priority(5, 7).unwrap();
        plic.set_enable_word(0, 0, (1 << 4) | (1 << 5)).unwrap();
        pulse_interrupt(&mut plic, 4);
        pulse_interrupt(&mut plic, 5);

        assert_eq!(plic.update_context_irq_line(0), Some(4));
        assert_eq!(plic.get_claim_complete(0).unwrap(), 4);
    }

    #[test]
    fn lowering_source_does_not_retract_an_accepted_request() {
        let mut plic = PLIC::new();
        plic.set_priority(6, 5).unwrap();
        plic.set_enable_word(0, 0, 1 << 6).unwrap();

        plic.set_interrupt_level(6, true);
        plic.set_interrupt_level(6, false);

        assert!(plic.get_pending_bit(6).unwrap());
        assert_eq!(plic.get_claim_complete(0).unwrap(), 6);
        plic.set_claim_complete(0, 6).unwrap();
        assert_eq!(plic.get_claim_complete(0).unwrap(), 0);
    }

    #[test]
    fn high_source_reasserts_once_after_completion() {
        let mut plic = PLIC::new();
        plic.set_priority(7, 5).unwrap();
        plic.set_enable_word(0, 0, 1 << 7).unwrap();

        plic.set_interrupt_level(7, true);
        plic.set_interrupt_level(7, true);
        assert_eq!(plic.get_claim_complete(0).unwrap(), 7);

        // Repeated high notifications while claimed must not create another
        // pending request before completion.
        plic.set_interrupt_level(7, true);
        assert!(!plic.get_pending_bit(7).unwrap());

        plic.set_claim_complete(0, 7).unwrap();
        assert!(plic.get_pending_bit(7).unwrap());
        assert_eq!(plic.get_claim_complete(0).unwrap(), 7);

        plic.set_interrupt_level(7, false);
        plic.set_claim_complete(0, 7).unwrap();
        assert_eq!(plic.get_claim_complete(0).unwrap(), 0);
    }

    struct TestPlicDevice {
        level: bool,
        samples: Cell<usize>,
    }

    impl TestPlicDevice {
        fn new(level: bool) -> Self {
            Self {
                level,
                samples: Cell::new(0),
            }
        }
    }

    impl DeviceTrait for TestPlicDevice {
        fn read(&mut self, _addr: WordType, _len: u32) -> Result<u64, MemError> {
            Err(MemError::LoadFault)
        }

        fn write(&mut self, _addr: WordType, _len: u32, _data: u64) -> Result<(), MemError> {
            Err(MemError::StoreFault)
        }

        fn sync(&mut self) {}
    }

    impl PlicDevice for TestPlicDevice {
        fn irq_level(&mut self) -> bool {
            self.samples.set(self.samples.get() + 1);
            self.level
        }
    }

    #[test]
    fn completion_samples_device_after_arena_registration() {
        let mut plic = PLIC::new();
        plic.set_priority(10, 5).unwrap();
        plic.set_enable_word(0, 0, 1 << 10).unwrap();

        let mut device = Box::new(TestPlicDevice::new(false));
        let device_ptr = NonNull::from(&mut *device as &mut dyn PlicDevice);
        plic.register_device(device_ptr, 10);
        let mut arena_builder = DeviceArenaBuilder::new();
        let device = arena_builder.register(device);
        let mut arena = arena_builder.build();

        assert!(plic.update_context_irq_line(0).is_none());

        arena.device_mut(device).level = true;
        assert_eq!(plic.update_context_irq_line(0), Some(10));
        assert_eq!(plic.get_claim_complete(0).unwrap(), 10);

        // The device clears its line before the guest writes complete. The
        // complete path must sample that state before deciding whether to
        // create another pending request.
        arena.device_mut(device).level = false;
        plic.set_claim_complete(0, 10).unwrap();
        assert!(!plic.get_pending_bit(10).unwrap());
        assert_eq!(plic.get_claim_complete(0).unwrap(), 0);
    }

    struct RecordingCPUInterruptHandler {
        levels: Vec<bool>,
    }

    impl crate::board::virt::RiscvIRQHandler for RecordingCPUInterruptHandler {
        fn handle_irq(&mut self, _interrupt: crate::isa::riscv::trap::Interrupt, level: bool) {
            self.levels.push(level);
        }
    }

    #[test]
    fn context_irq_line_is_driven_only_on_level_changes() {
        use crate::{
            board::virt::{IRQLine, RiscvIRQSource},
            isa::riscv::trap::Interrupt,
        };

        let mut cpu_handler = RecordingCPUInterruptHandler { levels: Vec::new() };
        let mut plic = PLIC::new();
        plic.set_irq_line(
            IRQLine::new(&mut cpu_handler, Interrupt::MachineExternal),
            0,
        );
        plic.set_priority(11, 5).unwrap();
        plic.set_enable_word(0, 0, 1 << 11).unwrap();

        plic.set_interrupt_level(11, true);
        assert_eq!(plic.update_context_irq_line(0), Some(11));
        assert_eq!(plic.update_context_irq_line(0), Some(11));
        assert_eq!(cpu_handler.levels, vec![false, true]);

        assert_eq!(plic.get_claim_complete(0).unwrap(), 11);
        assert!(plic.update_context_irq_line(0).is_none());
        assert_eq!(cpu_handler.levels, vec![false, true, false]);
    }

    #[test]
    fn multiple_contexts_share_one_peripheral_snapshot() {
        let mut plic = PLIC::new();
        let mut device = Box::new(TestPlicDevice::new(false));
        let device_ptr = NonNull::from(&mut *device as &mut dyn PlicDevice);
        plic.register_device(device_ptr, 12);
        let mut arena_builder = DeviceArenaBuilder::new();
        let device = arena_builder.register(device);
        let arena = arena_builder.build();

        plic.update_context_irq_lines(&[0, 1]);

        assert_eq!(arena.device(device).samples.get(), 1);
    }
}
