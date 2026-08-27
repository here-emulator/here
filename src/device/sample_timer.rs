#![allow(unused)]
#![cfg(feature = "test-device")]

//! A simple millisecond timer used to exercise external interrupts.

use std::{
    hint::cold_path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use tokio::{sync::watch, time::Instant};

use crate::{
    config::arch_config::WordType,
    device::{
        DeviceTrait, MemError, MemMappedDeviceTrait, PlicDevice,
        config::{SAMPLE_TIMER_BASE, SAMPLE_TIMER_SIZE},
        plic::PeriphIrqId,
    },
    task_spawner::DeviceTask,
};

pub const SAMPLE_TIMER_INTERRUPT_ID: PeriphIrqId = 63;
const CONTROL_RESET: u32 = 1 << 0;

struct SampleTimerLayout {
    control_register: u32,
    interrupt_mask_reg: u32,
    interval_low: u32,
    interval_high: u32,
}

impl SampleTimerLayout {
    fn new() -> Self {
        Self {
            control_register: 0,
            interrupt_mask_reg: 0,
            interval_low: 0,
            interval_high: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum TimerCommand {
    Schedule(Instant),
    Cancel,
}

pub struct SampleTimerDevice {
    layout: SampleTimerLayout,
    irq_pending: Arc<AtomicBool>,
    sender: watch::Sender<TimerCommand>,
}

pub struct TimerTask {
    rx: watch::Receiver<TimerCommand>,
    irq_pending: Arc<AtomicBool>,
}

impl DeviceTask for TimerTask {
    async fn run_simple(mut self) {
        let mut command = TimerCommand::Cancel;

        loop {
            match command {
                TimerCommand::Schedule(deadline) => tokio::select! {
                    result = self.rx.changed() => {
                        if result.is_err() {
                            return;
                        }
                        command = *self.rx.borrow();
                    }

                    _ = tokio::time::sleep_until(deadline) => {
                        self.irq_pending.store(true, Ordering::Release);
                        command = TimerCommand::Cancel;
                    }
                },
                TimerCommand::Cancel => {
                    if self.rx.changed().await.is_err() {
                        return;
                    }
                    command = *self.rx.borrow();
                }
            }
        }
    }
}

impl SampleTimerDevice {
    pub fn new() -> (Self, TimerTask) {
        let (tx, rx) = watch::channel(TimerCommand::Cancel);
        let irq_pending = Arc::new(AtomicBool::new(false));

        (
            Self {
                layout: SampleTimerLayout::new(),
                irq_pending: irq_pending.clone(),
                sender: tx,
            },
            TimerTask { rx, irq_pending },
        )
    }

    fn get_interval(&self) -> Duration {
        let data = (self.layout.interval_high as u64) << 32 | (self.layout.interval_low as u64);
        Duration::from_millis(data)
    }

    fn reset_timer(&mut self) {
        self.sender
            .send(TimerCommand::Schedule(Instant::now() + self.get_interval()))
            .unwrap();
    }
}

impl DeviceTrait for SampleTimerDevice {
    fn read(&mut self, addr: WordType, len: u32) -> Result<u64, MemError> {
        if len != 4 {
            cold_path();
            return Err(crate::device::MemError::LoadFault);
        }

        let data = match addr {
            0x00 => self.layout.control_register,
            0x04 => self.layout.interrupt_mask_reg,
            0x08 => self.layout.interval_low,
            0x0c => self.layout.interval_high,
            _ => return Err(MemError::LoadFault),
        };
        Ok(data as u64)
    }

    fn write(&mut self, addr: WordType, len: u32, data: u64) -> Result<(), MemError> {
        if len != 4 {
            cold_path();
            return Err(crate::device::MemError::StoreFault);
        }

        let data = data as u32;

        match addr {
            0x00 => {
                self.layout.control_register = data;
                if data & CONTROL_RESET != 0 {
                    // Deassert synchronously so a following PLIC completion cannot
                    // observe the stale interrupt level before the worker runs.
                    self.irq_pending.store(false, Ordering::Release);
                    self.reset_timer();
                }
            }
            0x04 => {
                self.layout.interrupt_mask_reg = data;
            }
            0x08 => {
                self.layout.interval_low = data;
                self.reset_timer();
            }
            0x0c => {
                self.layout.interval_high = data;
                self.reset_timer();
            }
            _ => return Err(MemError::StoreFault),
        };
        return Ok(());
    }

    fn sync(&mut self) {
        // nothing to do.
    }
}

impl MemMappedDeviceTrait for SampleTimerDevice {
    fn base() -> WordType {
        SAMPLE_TIMER_BASE
    }
    fn size() -> WordType {
        SAMPLE_TIMER_SIZE
    }
}

impl PlicDevice for SampleTimerDevice {
    fn irq_level(&mut self) -> bool {
        self.irq_pending.load(Ordering::Acquire) && (self.layout.interrupt_mask_reg & 1) == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn wait_for_pending(irq_pending: &AtomicBool) {
        tokio::time::timeout(Duration::from_secs(1), async {
            while !irq_pending.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("sample timer interrupt timed out");
    }

    #[tokio::test]
    async fn timer_task_rearms_after_a_new_schedule() {
        let (device, timer_task) = SampleTimerDevice::new();
        let sender = device.sender.clone();
        let irq_pending = device.irq_pending.clone();
        let task = tokio::spawn(timer_task.run_simple());

        sender
            .send(TimerCommand::Schedule(
                Instant::now() + Duration::from_millis(1),
            ))
            .unwrap();
        wait_for_pending(&irq_pending).await;
        assert!(irq_pending.load(Ordering::Acquire));

        irq_pending.store(false, Ordering::Release);
        sender
            .send(TimerCommand::Schedule(
                Instant::now() + Duration::from_millis(1),
            ))
            .unwrap();
        wait_for_pending(&irq_pending).await;
        assert!(irq_pending.load(Ordering::Acquire));

        drop(sender);
        drop(device);
        task.await.unwrap();
    }

    #[tokio::test]
    async fn timer_task_applies_schedule_that_interrupts_active_deadline() {
        let (device, timer_task) = SampleTimerDevice::new();
        let sender = device.sender.clone();
        let irq_pending = device.irq_pending.clone();
        let task = tokio::spawn(timer_task.run_simple());

        sender
            .send(TimerCommand::Schedule(
                Instant::now() + Duration::from_secs(60),
            ))
            .unwrap();
        tokio::task::yield_now().await;

        sender
            .send(TimerCommand::Schedule(
                Instant::now() + Duration::from_millis(1),
            ))
            .unwrap();
        wait_for_pending(&irq_pending).await;

        drop(sender);
        drop(device);
        task.await.unwrap();
    }

    #[tokio::test]
    async fn timer_task_cancel_keeps_the_task_available_for_rescheduling() {
        let (device, timer_task) = SampleTimerDevice::new();
        let sender = device.sender.clone();
        let irq_pending = device.irq_pending.clone();
        let task = tokio::spawn(timer_task.run_simple());

        sender
            .send(TimerCommand::Schedule(
                Instant::now() + Duration::from_millis(50),
            ))
            .unwrap();
        sender.send(TimerCommand::Cancel).unwrap();
        tokio::time::sleep(Duration::from_millis(75)).await;
        assert!(!irq_pending.load(Ordering::Acquire));

        sender
            .send(TimerCommand::Schedule(
                Instant::now() + Duration::from_millis(1),
            ))
            .unwrap();
        wait_for_pending(&irq_pending).await;

        drop(sender);
        drop(device);
        task.await.unwrap();
    }

    #[test]
    fn control_bit_zero_does_not_clear_pending_interrupt() {
        let (mut device, _task) = SampleTimerDevice::new();
        device.irq_pending.store(true, Ordering::Release);

        device.write_u32(0, !CONTROL_RESET).unwrap();

        assert!(device.irq_pending.load(Ordering::Acquire));
    }

    #[test]
    fn control_bit_one_clears_pending_interrupt_immediately() {
        let (mut device, _task) = SampleTimerDevice::new();
        device.irq_pending.store(true, Ordering::Release);

        device.write_u32(0, CONTROL_RESET).unwrap();

        assert!(!device.irq_pending.load(Ordering::Acquire));
    }
}
