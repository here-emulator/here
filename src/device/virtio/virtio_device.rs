use std::sync::{
    Mutex,
    atomic::{AtomicU8, AtomicU16},
};

use lazy_static::lazy_static;

use crate::device::virtio::config::VirtIOFeatureSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
#[allow(unused)]
pub enum VirtIODeviceEnum {
    Reserved = 0,
    VirtIONet = 1,
    VirtIOBlock = 2,
    VirtIOConsole = 3,
    VirtIORng = 4,
    VirtIOBalloonTraditional = 5,
    VirtIOIomem = 6,
    VirtIORpmsg = 7,
    VirtIOScsi = 8,
    VirtIO9p = 9,
    VirtIOMac80211Wlan = 10,
    VirtIORprocSerial = 11,
    VirtIOCaif = 12,
    VirtIOBalloon = 13,
    VirtIOGpu = 16,
    VirtIOClock = 17,
    VirtIOInput = 18,
    VirtIOVsock = 19,
    VirtIOCrypto = 20,
    VirtIOSignalDistribution = 21,
    VirtIOPstore = 22,
    VirtIOIommu = 23,
    VirtIOMem = 24,
    VirtIOSound = 25,
    VirtIOFs = 26,
    VirtIOPmem = 27,
    VirtIORpmb = 28,
    VirtIOMac80211Hwsim = 29,
    VirtIOVideoEncoder = 30,
    VirtIOVideoDecoder = 31,
    VirtIOScmi = 32,
    VirtIONitroSecureModule = 33,
    VirtIOI2cAdapter = 34,
    VirtIOWatchdog = 35,
    VirtIOCan = 36,
    VirtIOParameterServer = 38,
    VirtIOAudioPolicy = 39,
    VirtIOBluetooth = 40,
    VirtIOGpio = 41,
    VirtIORdma = 42,
    VirtIOCamera = 43,
    VirtIOIsm = 44,
    VirtIOSpiMaster = 45,
}

pub(crate) trait VirtIODeviceTrait {
    fn get_device_id(&self) -> u16;
    fn status(&mut self) -> &mut u8;
    fn get_generation(&self) -> u32;
    fn reset(&mut self);

    fn isr(&self) -> &AtomicU8;
    fn irq_level(&mut self) -> bool;

    fn get_host_feature(&self) -> VirtIOFeatureSet;
    fn set_feature(&mut self, feature: VirtIOFeatureSet);

    fn set_queue_num(&mut self, num: u32);
    fn queue_ready(&self) -> bool;
    fn queue_select(&self, idx: u32);
    fn get_num_of_queue(&self) -> u32; // device may have queue more than one.

    fn set_desc(&mut self, addr: u64);
    fn set_avail(&mut self, addr: u64);
    fn set_used(&mut self, addr: u64);

    fn manage_one_request(&mut self) -> bool;
    fn notify(&mut self, queue_idx: u32);

    fn read_config(&mut self, offset: u64, len: u32) -> u64;
    fn write_config(&mut self, offset: u64, len: u32, data: u64);
}

pub(super) struct DeviceIDAllocator(AtomicU16);

impl DeviceIDAllocator {
    pub(super) fn new() -> Self {
        Self(AtomicU16::new(0))
    }
    pub(super) fn alloc(&mut self) -> u16 {
        self.0.fetch_add(1, std::sync::atomic::Ordering::AcqRel)
    }
}

lazy_static! {
    pub(super) static ref DEVICE_ID_ALLOCTOR: Mutex<DeviceIDAllocator> =
        Mutex::new(DeviceIDAllocator::new());
}
