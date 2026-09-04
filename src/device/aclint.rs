use std::{cell::UnsafeCell, rc::Rc};

use crate::{
    board::virt::{IRQLine, RiscvIRQSource},
    clock::{Clock, Timer, VirtualClock},
    config::arch_config::WordType,
    device::{DeviceTrait, MemError},
    utils::{concat_to_u64, negative_of},
};

/// MMIO region size for the CLINT device.
pub const CLINT_SIZE: WordType = 0x10000;

pub struct Clint {
    hart_num: u32,
    msip_base: u64,
    time_base: u64,
    timecmp_base: u64,
    time_offset: u64,
    clock: VirtualClock,
    timer: Rc<UnsafeCell<Timer<VirtualClock>>>,
    msip: Vec<u32>,
    time_cmp: Vec<u64>,
    timer_irq_line: Option<IRQLine>, // TODO: Current implementation only supports 1 hart.
    software_irq_line: Option<IRQLine>,
    timer_cb_id: u64,
}

impl Clint {
    pub fn new(
        hart_num: u32,
        msip_base: u64,
        mtime_base: u64,
        mtimecmp_base: u64,
        clock: VirtualClock,
        timer: Rc<UnsafeCell<Timer<VirtualClock>>>,
    ) -> Self {
        Self {
            hart_num,
            msip_base,
            time_base: mtime_base,
            timecmp_base: mtimecmp_base,
            time_offset: negative_of(clock.now()),
            clock,
            timer,
            msip: vec![0u32; hart_num as usize],
            time_cmp: vec![0u64; hart_num as usize],
            timer_irq_line: None,
            software_irq_line: None,
            timer_cb_id: u64::MAX,
        }
    }
}

impl Clint {
    fn get_time(&mut self) -> u64 {
        self.clock.now().wrapping_add(self.time_offset)
    }

    fn update_timer(&mut self, hartid: usize) {
        if self.time_cmp[hartid] <= self.get_time() {
            self.timer_irq_line.as_mut().unwrap().set_irq(true);
        } else {
            self.timer_irq_line.as_mut().unwrap().set_irq(false);
            unsafe { self.timer.as_mut_unchecked() }.set_due(
                self.timer_cb_id,
                self.time_cmp[hartid].wrapping_sub(self.time_offset),
            );
        }
    }

    fn read_impl<T>(&mut self, addr: WordType) -> Result<T, MemError>
    where
        T: crate::utils::UnsignedInteger,
    {
        if self.msip_base <= addr && addr < self.msip_base + ((self.hart_num as u64) << 2) {
            let hartid = ((addr - self.msip_base) >> 2) as usize;
            if hartid >= self.msip.len() {
                return Err(MemError::LoadFault);
            }
            return Ok(T::truncate_from(self.msip[hartid]));
        } else if self.timecmp_base <= addr
            && addr < self.timecmp_base + ((self.hart_num as u64) << 3)
        {
            // timecmp
            let hartid = ((addr - self.timecmp_base) >> 3) as usize;

            if hartid >= self.time_cmp.len() {
                return Err(MemError::LoadFault);
            }

            if (addr & 0x7) == 0 {
                return Ok(T::truncate_from(self.time_cmp[hartid]));
            } else if (addr & 0x7) == 4 {
                // timecmp_hi in RV32
                return Ok(T::truncate_from(self.time_cmp[hartid] >> 32));
            } else {
                return Err(MemError::LoadFault);
            }
        } else if addr == self.time_base {
            return Ok(T::truncate_from(self.get_time()));
        } else if addr == self.time_base + 4 {
            // time_hi for RV32
            return Ok(T::truncate_from(self.get_time() >> 32));
        }

        Err(MemError::LoadFault)
    }

    fn write_impl<T>(&mut self, addr: WordType, data: T) -> Result<(), MemError>
    where
        T: crate::utils::UnsignedInteger,
    {
        if self.msip_base <= addr && addr < self.msip_base + ((self.hart_num as u64) << 2) {
            let hartid = ((addr - self.msip_base) >> 2) as usize;
            if hartid >= self.msip.len() {
                return Err(MemError::StoreFault);
            }
            let val: u32 = data.truncate_to();
            self.msip[hartid] = val;

            if let Some(irq) = &mut self.software_irq_line {
                irq.set_irq((val & 1) != 0);
            }
            Ok(())
        } else if self.timecmp_base <= addr
            && addr < self.timecmp_base + ((self.hart_num as u64) << 3)
        {
            // timecmp
            let hartid = ((addr - self.timecmp_base) >> 3) as usize;

            if hartid >= self.time_cmp.len() {
                return Err(MemError::StoreFault);
            }

            if (addr & 0x7) == 0 {
                self.time_cmp[hartid] = data.truncate_to();
                self.update_timer(hartid);
            } else if (addr & 0x7) == 4 {
                let timecmp_lo = (self.time_cmp[hartid] & 0xffffffff) as u32;
                let timecmp_hi: u32 = data.truncate_to();
                self.time_cmp[hartid] = concat_to_u64(timecmp_hi, timecmp_lo);
                self.update_timer(hartid);
            }
            Ok(())
        } else if addr == self.time_base || addr == self.time_base + 4 {
            // mtime
            let curr_clocktime = self.clock.now();
            let prev_mtime = self.get_time();

            if addr == self.time_base {
                let value = if T::BITS == 32 {
                    let time_hi = (prev_mtime >> 32) as u32;
                    let time_lo: u32 = data.truncate_to();
                    concat_to_u64(time_hi, time_lo)
                } else {
                    data.truncate_to()
                };

                // To make new `mtime` value == curr_clocktime + time_offset (mod 2^64)
                self.time_offset = value.wrapping_sub(curr_clocktime);
            } else {
                // addr == self.time_base + 4, write to `mtime_hi`
                let time_lo = (prev_mtime & 0xffff_ffff) as u32;
                let time_hi: u32 = data.truncate_to();
                let value = concat_to_u64(time_hi, time_lo);
                self.time_offset = value.wrapping_sub(curr_clocktime);
            }

            for i in 0..self.hart_num as usize {
                self.update_timer(i);
            }
            Ok(())
        } else {
            Err(MemError::StoreFault)
        }
    }
}

impl RiscvIRQSource for Clint {
    fn set_irq_line(&mut self, line: IRQLine, id: usize) {
        if id == 0 {
            self.timer_irq_line = Some(line);
            self.timer_cb_id = unsafe { self.timer.as_mut_unchecked() }.register({
                let irq_line_ptr = self.timer_irq_line.as_mut().unwrap() as *mut IRQLine;
                move || {
                    unsafe { &mut *irq_line_ptr }.set_irq(true);
                }
            });
        } else if id == 1 {
            self.software_irq_line = Some(line);
        }
    }
}

impl DeviceTrait for Clint {
    dispatch_read_write! { read_impl, write_impl }

    fn sync(&mut self) {
        // Nothing to do
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::virt::RiscvIRQHandler;
    use crate::isa::riscv::trap::Interrupt;

    struct MockIrqHandler {
        triggered: bool,
        level: bool,
    }

    impl RiscvIRQHandler for MockIrqHandler {
        fn handle_irq(&mut self, _interrupt: Interrupt, level: bool) {
            self.triggered = true;
            self.level = level;
        }
    }

    fn create_test_clint() -> (
        Clint,
        Rc<UnsafeCell<Timer<VirtualClock>>>,
        Box<MockIrqHandler>,
        Box<MockIrqHandler>,
    ) {
        use std::cell::Cell;

        let cycles = Rc::new(Cell::new(0));
        let clock = VirtualClock::new(cycles);
        let timer = Rc::new(UnsafeCell::new(Timer::new(clock.clone())));
        let mut clint = Clint::new(1, 0x02000000, 0x0200bff8, 0x02004000, clock, timer.clone());

        let mut time_handler = Box::new(MockIrqHandler {
            triggered: false,
            level: false,
        });
        let irq_line = IRQLine::new(
            &mut *time_handler as *mut MockIrqHandler,
            Interrupt::MachineTimer,
        );
        clint.set_irq_line(irq_line, 0);

        let mut soft_handler = Box::new(MockIrqHandler {
            triggered: false,
            level: false,
        });
        let irq_line = IRQLine::new(
            &mut *soft_handler as *mut MockIrqHandler,
            Interrupt::MachineSoft,
        );
        clint.set_irq_line(irq_line, 1);

        (clint, timer, time_handler, soft_handler)
    }

    #[test]
    fn test_mtime_read_write() {
        let (mut clint, _timer, _time_handler, _soft_handler) = create_test_clint();

        let initial_time: u64 = clint.read_impl(0x0200bff8).unwrap();
        assert_eq!(initial_time, 0);

        // 测试写入 MTIME 寄存器 (32位低位)
        clint.write_impl::<u32>(0x0200bff8, 0x12345678).unwrap();
        let time_low: u32 = clint.read_impl(0x0200bff8).unwrap();
        assert_eq!(time_low, 0x12345678);

        // 测试写入 MTIME 寄存器 (32位高位)
        clint.write_impl::<u32>(0x0200bffc, 0x87654321).unwrap();
        let time_high: u32 = clint.read_impl(0x0200bffc).unwrap();
        assert_eq!(time_high, 0x87654321);

        let full_time: u64 = clint.read_impl(0x0200bff8).unwrap();
        assert_eq!(full_time, 0x8765432112345678);
    }

    #[test]
    fn test_mtimecmp_read_write() {
        let (mut clint, _timer, _time_handler, _soft_handler) = create_test_clint();

        // 测试读取 MTIMECMP 寄存器 (Hart 0)
        let initial_timecmp: u64 = clint.read_impl(0x02004000).unwrap();
        assert_eq!(initial_timecmp, 0);

        // 测试写入 MTIMECMP 寄存器 (64位)
        clint
            .write_impl::<u64>(0x02004000, 0x123456789abcdef0)
            .unwrap();
        let timecmp: u64 = clint.read_impl(0x02004000).unwrap();
        assert_eq!(timecmp, 0x123456789abcdef0);

        // 测试分别写入高低32位
        clint.write_impl::<u32>(0x02004000, 0xdeadbeef).unwrap(); // 低32位
        clint.write_impl::<u32>(0x02004004, 0xcafebabe).unwrap(); // 高32位

        let timecmp_low: u32 = clint.read_impl(0x02004000).unwrap();
        let timecmp_high: u32 = clint.read_impl(0x02004004).unwrap();
        assert_eq!(timecmp_low, 0xdeadbeef);
        assert_eq!(timecmp_high, 0xcafebabe);

        let result: Result<u32, _> = clint.read_impl(0x12345678);
        assert_eq!(result, Err(MemError::LoadFault));

        let result = clint.write_impl::<u32>(0x12345678, 0xdeadbeef);
        assert_eq!(result, Err(MemError::StoreFault));
    }

    #[test]
    fn test_msip_read_write() {
        let (mut clint, _timer, _time_handler, mut soft_handler) = create_test_clint();

        // Read MSIP (Hart 0)
        let initial_msip: u32 = clint.read_impl(0x02000000).unwrap();
        assert_eq!(initial_msip, 0);

        // Write MSIP (Hart 0)
        clint.write_impl::<u32>(0x02000000, 1).unwrap();
        let msip: u32 = clint.read_impl(0x02000000).unwrap();
        assert_eq!(msip, 1);
        assert!(soft_handler.triggered);
        assert!(soft_handler.level);

        // Clear MSIP (Hart 0)
        soft_handler.triggered = false;
        clint.write_impl::<u32>(0x02000000, 0).unwrap();
        let msip: u32 = clint.read_impl(0x02000000).unwrap();
        assert_eq!(msip, 0);
        assert!(soft_handler.triggered);
        assert!(!soft_handler.level);

        // Write MSIP (Hart 0) with other bits
        clint.write_impl::<u32>(0x02000000, 0x12345678).unwrap();
        let msip: u32 = clint.read_impl(0x02000000).unwrap();
        assert_eq!(msip, 0x12345678);
    }
}
