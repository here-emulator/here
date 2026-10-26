use crate::{config::arch_config::WordType, device::plic::PeriphIrqId};

#[derive(Debug, Clone)]
pub struct MemMapInfo {
    pub name: String,
    pub base: WordType,
    pub size: WordType,
    pub irq: Option<PeriphIrqId>,
}

#[derive(Debug, Clone)]
pub struct IdAllocator {
    id: WordType,
    device_name: String,
    mem_base: WordType,
    mem_size: WordType,
    irq_base: Option<PeriphIrqId>,
}

impl IdAllocator {
    pub fn new(
        start_id: WordType,
        device_name: String,
        mem_base: WordType,
        mem_size: WordType,
        irq_base: Option<PeriphIrqId>,
    ) -> Self {
        Self {
            id: start_id,
            device_name,
            mem_base,
            mem_size,
            irq_base,
        }
    }

    pub fn get(&mut self) -> MemMapInfo {
        let name = self.id.to_string() + &self.device_name;
        let base = self.mem_base + self.id * self.mem_size;
        let irq = self
            .irq_base
            .map(|base_irq| base_irq + self.id as PeriphIrqId);
        self.id += 1;
        MemMapInfo {
            name,
            base,
            size: self.mem_size,
            irq,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_allocator_with_irq() {
        let mut allocator = IdAllocator::new(0, "test_dev".to_string(), 0x1000, 0x100, Some(10));
        let info0 = allocator.get();
        assert_eq!(info0.name, "0test_dev");
        assert_eq!(info0.base, 0x1000);
        assert_eq!(info0.size, 0x100);
        assert_eq!(info0.irq, Some(10));

        let info1 = allocator.get();
        assert_eq!(info1.name, "1test_dev");
        assert_eq!(info1.base, 0x1100);
        assert_eq!(info1.size, 0x100);
        assert_eq!(info1.irq, Some(11));
    }

    #[test]
    fn test_id_allocator_without_irq() {
        let mut allocator = IdAllocator::new(1, "no_irq".to_string(), 0x2000, 0x80, None);
        let info = allocator.get();
        assert_eq!(info.name, "1no_irq");
        assert_eq!(info.base, 0x2000 + 1 * 0x80);
        assert_eq!(info.size, 0x80);
        assert_eq!(info.irq, None);
    }
}
