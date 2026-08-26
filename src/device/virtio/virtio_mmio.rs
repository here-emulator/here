use std::cell::UnsafeCell;

use bitflags::bitflags;
use log::{error, info};
use num_enum::TryFromPrimitive;

use crate::{
    device::{
        DeviceTrait, MemError, MemMappedDeviceTrait, PlicDevice,
        config::{VIRTIO_MMIO_BASE, VIRTIO_MMIO_SIZE},
        virtio::{
            config::*,
            virtio_device::{VirtIODeviceEnum, VirtIODeviceTrait},
        },
    },
    utils::check_align,
};

pub const VIRTIO_DEVICE_ID_BLOCK: u16 = VirtIODeviceEnum::VirtIOBlock as u16;
const VIRTIO_MMIO_CONFIG_OFFSET: u64 = VirtIO_MMIO_Offset::Config as u64;
const VIRTIO_MMIO_MAX_QUEUES: usize = 8;
const FEATURE_SELECT_BITS: u32 = 32;
const FEATURE_WORD_COUNT: u32 = (VIRTIO_MAX_FEATURE_BIT_LEN / FEATURE_SELECT_BITS as usize) as u32;

#[repr(u64)]
#[derive(TryFromPrimitive, Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum VirtIO_MMIO_Offset {
    MagicValue = 0x000,
    Version = 0x004,
    DeviceId = 0x008,
    VendorId = 0x00C,
    DeviceFeatures = 0x010,
    DeviceFeaturesSelect = 0x014,
    DriverFeatures = 0x020,
    DriverFeaturesSelect = 0x024,
    QueueSelect = 0x030,
    QueueNumMax = 0x034,
    QueueNum = 0x038,
    QueueReady = 0x044,
    QueueNotify = 0x050,
    InterruptStatus = 0x060,
    InterruptAck = 0x064,
    Status = 0x070,
    QueueDescLow = 0x080,
    QueueDescHigh = 0x084,
    QueueAvailLow = 0x090,  // Driver ring
    QueueAvailHigh = 0x094, // Driver ring
    QueueUsedLow = 0x0A0,   // Device ring
    QueueUsedHigh = 0x0A4,  // Device ring
    SharedMemSelect = 0x0AC,
    SharedMemLenLow = 0x0B0,
    SharedMemLenHigh = 0x0B4,
    SharedMemBaseLow = 0x0B8,
    SharedMemBaseHigh = 0x0BC,
    // QueueReset = 0x0C0,
    ConfigGeneration = 0x0FC,
    Config = 0x100,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct VirtIODeviceStatus: u8 {
        /// Device acknowledges that it has seen the device and knows what it is.
        const ACKNOWLEDGE = 1 << 0;
        /// Driver has found a usable device.
        const DRIVER = 1 << 1;
        /// Driver has set up the device.
        const DRIVER_OK = 1 << 2;
        /// Driver has failed to set up the device or device has encountered an error.
        const FAILED = 1 << 7;
        /// Indicates that the driver is set up and ready to drive the device
        const FEATURES_OK = 1 << 3;
        /// Device needs to be reset.
        const DEVICE_NEEDS_RESET = 1 << 6;
    }
}

#[derive(Clone, Copy)]
struct VirtIOMMIOQueueStatus {
    num: u32,
    desc: u64,
    avail: u64,
    used: u64,
    enable: bool,
}
impl Default for VirtIOMMIOQueueStatus {
    fn default() -> Self {
        Self {
            num: 0,
            desc: 0,
            avail: 0,
            used: 0,
            enable: false,
        }
    }
}

impl VirtIOMMIOQueueStatus {
    fn set_low(value: &mut u64, low: u32) {
        *value = (*value & !u32::MAX as u64) | low as u64;
    }

    fn set_high(value: &mut u64, high: u32) {
        *value = (*value & u32::MAX as u64) | ((high as u64) << FEATURE_SELECT_BITS);
    }
}

pub(crate) struct VirtIOMMIO {
    device: Box<UnsafeCell<dyn VirtIODeviceTrait>>,
    host_features_sel: u32,
    guest_features_sel: u32,
    guest_features: VirtIOFeatureSet,

    queues: [VirtIOMMIOQueueStatus; VIRTIO_MMIO_MAX_QUEUES],
    queue_select: usize,
}

impl VirtIOMMIO {
    pub fn new(device: Box<UnsafeCell<dyn VirtIODeviceTrait>>) -> Self {
        Self {
            device,
            host_features_sel: 0,
            guest_features_sel: 0,
            guest_features: 0,

            queues: [VirtIOMMIOQueueStatus::default(); VIRTIO_MMIO_MAX_QUEUES],
            queue_select: 0,
        }
    }

    fn selected_queue(&self) -> Option<&VirtIOMMIOQueueStatus> {
        self.queues.get(self.queue_select)
    }

    fn selected_queue_mut(&mut self) -> Option<&mut VirtIOMMIOQueueStatus> {
        self.queues.get_mut(self.queue_select)
    }

    fn reset(&mut self) {
        self.host_features_sel = 0;
        self.guest_features_sel = 0;
        self.guest_features = 0;
        self.queues = [VirtIOMMIOQueueStatus::default(); VIRTIO_MMIO_MAX_QUEUES];
        self.queue_select = 0;
        unsafe { self.device.as_mut_unchecked() }.reset();
    }

    fn feature_word(features: VirtIOFeatureSet, select: u32) -> u32 {
        if select >= FEATURE_WORD_COUNT {
            0
        } else {
            (features >> (select * FEATURE_SELECT_BITS)) as u32
        }
    }

    fn set_feature_word(features: &mut VirtIOFeatureSet, select: u32, value: u32) {
        if select >= FEATURE_WORD_COUNT {
            return;
        }

        let shift = select * FEATURE_SELECT_BITS;
        let mask = (u32::MAX as VirtIOFeatureSet) << shift;
        *features = (*features & !mask) | ((value as VirtIOFeatureSet) << shift);
    }

    fn read_config<T>(&mut self, offset: u64) -> Result<T, MemError>
    where
        T: crate::utils::UnsignedInteger,
    {
        if !check_align::<T>(offset) {
            return Err(MemError::LoadMisaligned);
        }

        let vdev = unsafe { self.device.as_mut_unchecked() };
        Ok(T::truncate_from(vdev.read_config(
            offset - VIRTIO_MMIO_CONFIG_OFFSET,
            size_of::<T>() as u32,
        )))
    }

    fn write_config<T>(&mut self, offset: u64, data: T) -> Result<(), MemError>
    where
        T: crate::utils::UnsignedInteger,
    {
        if !check_align::<T>(offset) {
            return Err(MemError::StoreMisaligned);
        }

        let vdev = unsafe { self.device.as_mut_unchecked() };
        vdev.write_config(
            offset - VIRTIO_MMIO_CONFIG_OFFSET,
            size_of::<T>() as u32,
            data.into(),
        );
        Ok(())
    }

    fn read_u32_impl(&self, offset: u64) -> u32 {
        let vdev = unsafe { self.device.as_mut_unchecked() };

        if !check_align::<u32>(offset) {
            // will be checked in mmio.
            unreachable!()
        }

        if offset >= VIRTIO_MMIO_CONFIG_OFFSET {
            return vdev.read_config(offset - VIRTIO_MMIO_CONFIG_OFFSET, size_of::<u32>() as u32)
                as u32;
        }

        let offset_type = VirtIO_MMIO_Offset::try_from(offset);
        match offset_type {
            Err(error) => {
                error!(
                    "VirtIO: read of unimplemented register: {:#x}, {}",
                    offset, error
                );
                0
            }
            Ok(offset_type) => {
                let ret = match  offset_type {
                    VirtIO_MMIO_Offset::MagicValue => VIRT_MAGIC,
                    VirtIO_MMIO_Offset::Version => VIRT_VERSION,
                    VirtIO_MMIO_Offset::DeviceId => vdev.get_device_id() as u32,
                    VirtIO_MMIO_Offset::VendorId => VIRT_VENDOR,
                    VirtIO_MMIO_Offset::DeviceFeatures => {
                        info!("virtio-mmio read device feature.");

                        Self::feature_word(vdev.get_host_feature(), self.host_features_sel)
                    }
                    VirtIO_MMIO_Offset::QueueNumMax => {
                        if self.queue_select < vdev.get_num_of_queue() as usize {
                            VIRTQUEUE_MAX_SIZE
                        } else {
                            0
                        }
                    }
                    // VirtIO_MMIO_Offset::QueuePFN => 0 as u32, // legacy
                    VirtIO_MMIO_Offset::QueueReady => {
                        self.selected_queue().is_some_and(|q| q.enable) as u32
                    }
                    VirtIO_MMIO_Offset::InterruptStatus => {
                        vdev.isr().load(std::sync::atomic::Ordering::Acquire) as u32
                    }
                    VirtIO_MMIO_Offset::Status => *vdev.status() as u32,
                    VirtIO_MMIO_Offset::ConfigGeneration => vdev.get_generation(),
                    VirtIO_MMIO_Offset::SharedMemLenLow | VirtIO_MMIO_Offset::SharedMemLenHigh => u32::MAX,
                    VirtIO_MMIO_Offset::Config => unreachable!(),
                    VirtIO_MMIO_Offset::DeviceFeaturesSelect
                    | VirtIO_MMIO_Offset::DriverFeatures
                    | VirtIO_MMIO_Offset::DriverFeaturesSelect
                    // | VirtIO_MMIO_Offset::GuestPageSize => 0 as u32, // legacy
                    | VirtIO_MMIO_Offset::QueueSelect
                    | VirtIO_MMIO_Offset::QueueNum
                    // | VirtIO_MMIO_Offset::QueueAligne => 0 as u32, // legacy
                    | VirtIO_MMIO_Offset::QueueNotify
                    | VirtIO_MMIO_Offset::InterruptAck
                    | VirtIO_MMIO_Offset::QueueDescLow
                    | VirtIO_MMIO_Offset::QueueDescHigh
                    | VirtIO_MMIO_Offset::QueueAvailLow
                    | VirtIO_MMIO_Offset::QueueAvailHigh
                    | VirtIO_MMIO_Offset::QueueUsedLow
                    | VirtIO_MMIO_Offset::QueueUsedHigh
                    => {
                        error!("VirtIO: read of write-only register: {:#x}", offset);
                        unreachable!()
                    }
                    // VirtIO_MMIO_Offset::QueueReset | 
                    VirtIO_MMIO_Offset::SharedMemBaseHigh | VirtIO_MMIO_Offset::SharedMemBaseLow | VirtIO_MMIO_Offset::SharedMemSelect => {
                        error!("VirtIO: read of unimplemented register: {:#x}", offset);
                        0
                    }
                };
                ret
            }
        }
    }

    fn write_u32_impl(&mut self, offset: u64, value: u32) {
        let vdev = self.device.get();

        if !check_align::<u32>(offset) {
            // will be checked in mmio.
            unreachable!()
        }

        if offset >= VIRTIO_MMIO_CONFIG_OFFSET {
            unsafe { &mut *vdev }.write_config(
                offset - VIRTIO_MMIO_CONFIG_OFFSET,
                size_of::<u32>() as u32,
                value as u64,
            );
            return;
        }

        let offset_type = VirtIO_MMIO_Offset::try_from(offset);
        match offset_type {
            Err(error) => {
                error!(
                    "VirtIO: write of unimplemented register: {:#x}, {}",
                    offset, error
                );
            }
            Ok(offset_type) => match offset_type {
                VirtIO_MMIO_Offset::DeviceFeaturesSelect => self.host_features_sel = value,
                VirtIO_MMIO_Offset::DriverFeatures => {
                    info!("virtio-mmio write driver feature.");
                    Self::set_feature_word(
                        &mut self.guest_features,
                        self.guest_features_sel,
                        value,
                    );
                }
                VirtIO_MMIO_Offset::DriverFeaturesSelect => {
                    self.guest_features_sel = value;
                }
                // VirtIO_MMIO_Offset::GUEST_PAGE_SIZE => {}, // legacy
                VirtIO_MMIO_Offset::QueueSelect => {
                    self.queue_select = value as usize;
                    unsafe { &mut *vdev }.queue_select(value);
                }
                VirtIO_MMIO_Offset::QueueNum => {
                    if value == 0 || value > VIRTQUEUE_MAX_SIZE {
                        error!(
                            "VirtIO: invalid queue size {value}, maximum is {VIRTQUEUE_MAX_SIZE}"
                        );
                        return;
                    }
                    let queue_select = self.queue_select;
                    if let Some(q) = self.selected_queue_mut() {
                        q.num = value;
                    }
                    let vdev = unsafe { &mut *vdev };
                    if queue_select < vdev.get_num_of_queue() as usize {
                        vdev.set_queue_num(value);
                    }
                }
                // VirtIO_MMIO_Offset::QueueAlign => {}, // legacy
                // VirtIO_MMIO_Offset::QueuePFN => {}, // legacy
                VirtIO_MMIO_Offset::QueueReady => {
                    if value > 1 {
                        error!("VirtIO: QueueReady only accepts 0 or 1, got {value}");
                        return;
                    }
                    let queue_select = self.queue_select;
                    let queue_config = if let Some(q) = self.selected_queue_mut() {
                        if value == 1 && q.num == 0 {
                            error!("VirtIO: cannot enable queue {queue_select} with size 0");
                            return;
                        }
                        q.enable = value != 0;
                        q.enable.then_some((q.num, q.desc, q.avail, q.used))
                    } else {
                        error!(
                            "VirtIO: QueueReady write for invalid queue {}",
                            queue_select
                        );
                        None
                    };
                    if let Some((num, desc, avail, used)) = queue_config {
                        let vdev = unsafe { &mut *vdev };
                        if queue_select < vdev.get_num_of_queue() as usize {
                            vdev.set_queue_num(num);
                            vdev.set_desc(desc);
                            vdev.set_avail(avail);
                            vdev.set_used(used);
                            if !vdev.queue_ready() {
                                error!("VirtIO: queue {queue_select} addresses are invalid");
                                self.queues[queue_select].enable = false;
                            }
                        }
                    };
                }
                VirtIO_MMIO_Offset::QueueNotify => {
                    // Publish the driver's queue writes before ringing the doorbell.
                    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                    let queue_idx = value;
                    let vdev = unsafe { &mut *vdev };
                    let ready = self
                        .queues
                        .get(queue_idx as usize)
                        .is_some_and(|queue| queue.enable);
                    if queue_idx < vdev.get_num_of_queue() && ready {
                        vdev.notify(queue_idx);
                    }
                }
                VirtIO_MMIO_Offset::InterruptAck => {
                    let vdev = unsafe { &mut *vdev };
                    let clear_mask = value as u8;
                    vdev.isr()
                        .fetch_and(!clear_mask, std::sync::atomic::Ordering::AcqRel);
                }
                VirtIO_MMIO_Offset::Status => {
                    if value == 0 {
                        self.reset();
                        return;
                    }
                    if let Some(new_status) = VirtIODeviceStatus::from_bits(value as u8) {
                        let vdev = unsafe { &mut *vdev };

                        *vdev.status() |= new_status.bits();
                        if !(new_status & VirtIODeviceStatus::FEATURES_OK).is_empty() {
                            vdev.set_feature(self.guest_features)
                        }
                    }
                }
                VirtIO_MMIO_Offset::QueueDescLow => {
                    if let Some(q) = self.selected_queue_mut() {
                        VirtIOMMIOQueueStatus::set_low(&mut q.desc, value);
                    }
                }
                VirtIO_MMIO_Offset::QueueDescHigh => {
                    if let Some(q) = self.selected_queue_mut() {
                        VirtIOMMIOQueueStatus::set_high(&mut q.desc, value);
                    }
                }
                VirtIO_MMIO_Offset::QueueAvailLow => {
                    if let Some(q) = self.selected_queue_mut() {
                        VirtIOMMIOQueueStatus::set_low(&mut q.avail, value);
                    }
                }
                VirtIO_MMIO_Offset::QueueAvailHigh => {
                    if let Some(q) = self.selected_queue_mut() {
                        VirtIOMMIOQueueStatus::set_high(&mut q.avail, value);
                    }
                }
                VirtIO_MMIO_Offset::QueueUsedLow => {
                    if let Some(q) = self.selected_queue_mut() {
                        VirtIOMMIOQueueStatus::set_low(&mut q.used, value);
                    }
                }
                VirtIO_MMIO_Offset::QueueUsedHigh => {
                    if let Some(q) = self.selected_queue_mut() {
                        VirtIOMMIOQueueStatus::set_high(&mut q.used, value);
                    }
                }
                VirtIO_MMIO_Offset::MagicValue
                | VirtIO_MMIO_Offset::Version
                | VirtIO_MMIO_Offset::DeviceId
                | VirtIO_MMIO_Offset::VendorId
                | VirtIO_MMIO_Offset::DeviceFeatures
                | VirtIO_MMIO_Offset::QueueNumMax
                | VirtIO_MMIO_Offset::InterruptStatus
                | VirtIO_MMIO_Offset::ConfigGeneration => {
                    error!("VirtIO: write to read-only register: {:#x}", offset);
                }
                VirtIO_MMIO_Offset::SharedMemBaseHigh
                | VirtIO_MMIO_Offset::SharedMemBaseLow
                | VirtIO_MMIO_Offset::SharedMemLenHigh
                | VirtIO_MMIO_Offset::SharedMemLenLow
                | VirtIO_MMIO_Offset::SharedMemSelect => {
                    error!("VirtIO: DO NOT allow shared memory.");
                }
                VirtIO_MMIO_Offset::Config => unreachable!(),
            },
        };
    }

    fn read_impl<T>(
        &mut self,
        addr: crate::config::arch_config::WordType,
    ) -> Result<T, crate::device::MemError>
    where
        T: crate::utils::UnsignedInteger,
    {
        if addr >= VIRTIO_MMIO_CONFIG_OFFSET {
            return self.read_config(addr);
        }

        if size_of::<T>() != size_of::<u32>() || !check_align::<u32>(addr) {
            return Err(MemError::LoadMisaligned);
        }

        let offset: u64 = addr;
        let val = self.read_u32_impl(offset);
        let val = unsafe { (&val as *const u32 as *const T).read() };
        Ok(val)
    }

    fn write_impl<T>(
        &mut self,
        addr: crate::config::arch_config::WordType,
        data: T,
    ) -> Result<(), crate::device::MemError>
    where
        T: crate::utils::UnsignedInteger,
    {
        if addr >= VIRTIO_MMIO_CONFIG_OFFSET {
            return self.write_config(addr, data);
        }

        if size_of::<T>() != size_of::<u32>() || !check_align::<u32>(addr) {
            return Err(MemError::StoreMisaligned);
        }

        let data = unsafe { (&data as *const T as *const u32).read() };
        let offset = addr;
        self.write_u32_impl(offset, data);
        Ok(())
    }
}

impl DeviceTrait for VirtIOMMIO {
    dispatch_read_write! { read_impl, write_impl }

    fn sync(&mut self) {}
}

impl MemMappedDeviceTrait for VirtIOMMIO {
    fn base() -> crate::config::arch_config::WordType {
        VIRTIO_MMIO_BASE
    }
    fn size() -> crate::config::arch_config::WordType {
        VIRTIO_MMIO_SIZE
    }
}

impl PlicDevice for VirtIOMMIO {
    fn irq_level(&mut self) -> bool {
        unsafe { self.device.as_mut_unchecked() }.irq_level()
    }
}

#[cfg(test)]
impl VirtIOMMIO {
    pub(crate) fn write_status(&mut self, status: VirtIODeviceStatus) {
        self.write_u32_impl(VirtIO_MMIO_Offset::Status as u64, status.bits() as u32);
    }

    pub(crate) fn get_host_feature(&mut self) -> VirtIOFeatureSet {
        let mut feature: VirtIOFeatureSet = 0;
        for i in (0..FEATURE_WORD_COUNT).rev() {
            feature <<= 32;
            self.write_u32_impl(VirtIO_MMIO_Offset::DeviceFeaturesSelect as u64, i);
            feature |=
                self.read_u32_impl(VirtIO_MMIO_Offset::DeviceFeatures as u64) as VirtIOFeatureSet;
        }
        feature
    }

    pub(crate) fn set_guest_feature(&mut self, mut feature: VirtIOFeatureSet) {
        for i in 0..FEATURE_WORD_COUNT {
            self.write_u32_impl(VirtIO_MMIO_Offset::DriverFeaturesSelect as u64, i);
            self.write_u32_impl(VirtIO_MMIO_Offset::DriverFeatures as u64, feature as u32);
            feature >>= 32;
        }
    }

    pub(crate) fn init_queue(&mut self, desc_base: u64, avail_base: u64, used_base: u64) {
        self.write_u32_impl(VirtIO_MMIO_Offset::QueueAvailLow as u64, avail_base as u32);
        self.write_u32_impl(
            VirtIO_MMIO_Offset::QueueAvailHigh as u64,
            (avail_base >> 32) as u32,
        );

        self.write_u32_impl(VirtIO_MMIO_Offset::QueueDescLow as u64, desc_base as u32);
        self.write_u32_impl(
            VirtIO_MMIO_Offset::QueueDescHigh as u64,
            (desc_base >> 32) as u32,
        );

        self.write_u32_impl(VirtIO_MMIO_Offset::QueueUsedLow as u64, used_base as u32);
        self.write_u32_impl(
            VirtIO_MMIO_Offset::QueueUsedHigh as u64,
            (used_base >> 32) as u32,
        );

        self.write_u32_impl(VirtIO_MMIO_Offset::QueueReady as u64, 0x01);
    }
}

#[cfg(test)]
pub(crate) struct GuestFeatureBuilder {
    feature: VirtIOFeatureSet,
}
#[cfg(test)]
impl GuestFeatureBuilder {
    pub(crate) fn new() -> Self {
        Self { feature: 0 }
    }

    pub(crate) fn add_guest_feature(mut self, one_feature: VirtIOFeatureSet) -> Self {
        self.feature |= one_feature;
        self
    }

    pub(crate) fn take(self) -> VirtIOFeatureSet {
        self.feature
    }
}

#[cfg(test)]
mod test {
    use core::slice;
    use std::io::{Read, Seek};

    use super::*;
    use crate::{
        device::{
            DeviceTrait, PlicDevice,
            virtio::{
                virtio_blk::{
                    VirtIOBlkDeviceBuilder, VirtIOBlkReqStatus, VirtIOBlockFeature, VirtioBlkReq,
                    VirtioBlkReqType, VirtioBlkStatus, init_block_file,
                },
                virtio_queue::{
                    VirtQueueAvail, VirtQueueAvailFlag, VirtQueueDesc, VirtQueueDescFlag,
                    VirtQueueUsed, VirtQueueUsedFlag,
                },
            },
        },
        ram::Ram,
        ram_config,
    };

    const QUEUE_NUM: usize = 8;
    const DESC_NUM: usize = 16;

    #[test]
    fn feature_registers_cover_all_128_bits() {
        let words = [0x0123_4567, 0x89ab_cdef, 0x1357_9bdf, 0x2468_ace0];
        let mut features = 0;

        for (selector, word) in words.into_iter().enumerate() {
            VirtIOMMIO::set_feature_word(&mut features, selector as u32, word);
        }
        for (selector, word) in words.into_iter().enumerate() {
            assert_eq!(VirtIOMMIO::feature_word(features, selector as u32), word);
        }

        assert_eq!(VirtIOMMIO::feature_word(features, FEATURE_WORD_COUNT), 0);
        let original = features;
        VirtIOMMIO::set_feature_word(&mut features, FEATURE_WORD_COUNT, u32::MAX);
        assert_eq!(features, original);
    }

    #[test]
    fn test_mmio_blk_device() {
        let file_name = String::from("./tmp/test_mmio_blk_device.txt");
        let mut buf: [u8; 512] = [0u8; 512];
        buf[0xff] = 0x55;
        let mut file = init_block_file(&file_name, 1, |_| &buf);
        let mut ram = Ram::new();
        let ram_base = &mut ram[0] as *mut u8;
        let virt_device = VirtIOBlkDeviceBuilder::new(ram_base, file_name)
            .name("VirtIO Block 0")
            .generation(0)
            .host_feature(VirtIOBlockFeature::BlockSize)
            .host_feature(VirtIOBlockFeature::Flush)
            .get();

        let mut virtio_mmio_device = VirtIOMMIO::new(Box::new(UnsafeCell::new(virt_device)));
        assert_eq!(
            virtio_mmio_device.read_u32_impl(VirtIO_MMIO_Offset::DeviceId as u64),
            VIRTIO_DEVICE_ID_BLOCK as u32
        );
        virtio_mmio_device.write_status(VirtIODeviceStatus::ACKNOWLEDGE);
        virtio_mmio_device.write_status(VirtIODeviceStatus::DRIVER);

        // set feature.
        let device_feature = virtio_mmio_device.get_host_feature();
        assert_ne!(device_feature & virtio_reserved_feature::VERSION_1, 0);
        let driver_feature = GuestFeatureBuilder::new()
            .add_guest_feature(virtio_reserved_feature::VERSION_1)
            .add_guest_feature(VirtIOBlockFeature::BlockSize.bit())
            .add_guest_feature(VirtIOBlockFeature::Flush.bit())
            .take();
        virtio_mmio_device.set_guest_feature(driver_feature);

        virtio_mmio_device.write_status(VirtIODeviceStatus::DRIVER_OK);
        virtio_mmio_device.write_status(VirtIODeviceStatus::FEATURES_OK);
        let status = virtio_mmio_device.read_u32_impl(VirtIO_MMIO_Offset::Status as u64);
        assert!(status & VirtIODeviceStatus::DRIVER_OK.bits() as u32 != 0);

        // init virt_queue.
        virtio_mmio_device.write_u32_impl(VirtIO_MMIO_Offset::QueueSelect as u64, 0);
        virtio_mmio_device.write_u32_impl(VirtIO_MMIO_Offset::QueueNum as u64, QUEUE_NUM as u32);

        let virtq_desc_base = 0x8000_2000 as u64;
        let virtq_avail_base = 0x8000_2100 + ((QUEUE_NUM + 2) * size_of::<u16>()) as u64;
        let virtq_used_base = 0x8000_2200 + (QUEUE_NUM * size_of::<VirtQueueUsed>() + 4) as u64;
        virtio_mmio_device.init_queue(virtq_desc_base, virtq_avail_base, virtq_used_base);

        // Description Table.
        let virt_queue_desc = unsafe {
            slice::from_raw_parts_mut(
                &mut ram[(virtq_desc_base - ram_config::BASE_ADDR) as usize] as *mut u8
                    as *mut VirtQueueDesc,
                DESC_NUM,
            )
        };

        // Available Ring.
        let virtq_avail = &mut ram[(virtq_avail_base - ram_config::BASE_ADDR) as usize] as *mut u8
            as *mut VirtQueueAvail;
        let virtq_avail = unsafe { virtq_avail.as_mut().unwrap() };
        virtq_avail.init(VirtQueueAvailFlag::Default);
        virtq_avail.idx_atomic_add(1);
        let avail_ring = VirtQueueAvail::mut_ring(virtq_avail as *mut _ as u64, QUEUE_NUM as u32);

        // Used Ring.
        let virtq_used = &mut ram[(virtq_used_base - ram_config::BASE_ADDR) as usize] as *mut u8
            as *mut VirtQueueUsed;
        let virtq_used = unsafe { virtq_used.as_mut().unwrap() };
        virtq_used.init(VirtQueueUsedFlag::Default);
        let _used_ring = virtq_used.ring(QUEUE_NUM as u32);

        // Write Available Ring.
        avail_ring[0] = 0;

        // header
        let desc0 = &mut virt_queue_desc[0];
        let desc0_buf_addr = 0x8000_2300;
        desc0.init(
            desc0_buf_addr,
            size_of::<VirtioBlkReq>() as u32,
            VirtQueueDescFlag::VIRTQ_DESC_F_NEXT,
            1,
        );
        let req = &mut ram[(desc0_buf_addr - ram_config::BASE_ADDR) as usize] as *mut u8
            as *mut VirtioBlkReq;
        let req = unsafe { req.as_mut().unwrap() };
        *req = VirtioBlkReq::new(VirtioBlkReqType::Out, 0);

        // data body
        let desc1 = &mut virt_queue_desc[1];
        let desc1_buf_addr = 0x8000_2400;
        desc1.init(
            desc1_buf_addr,
            0x200,
            VirtQueueDescFlag::VIRTQ_DESC_F_NEXT,
            2,
        );
        let desc_buf = unsafe {
            slice::from_raw_parts_mut(
                &mut ram[(desc1_buf_addr - ram_config::BASE_ADDR) as usize] as *mut u8,
                0x200,
            )
        };
        for i in 0..0x200 {
            desc_buf[i] = (i * i) as u8;
        }

        // result status
        let desc2 = &mut virt_queue_desc[2];
        let desc2_buf_addr = 0x8000_2310;
        desc2.init(
            0x8000_2310,
            size_of::<VirtioBlkStatus>() as u32, // 1 byte
            VirtQueueDescFlag::empty(),
            0,
        );
        let desc_status = unsafe {
            (&mut ram[(desc2_buf_addr - ram_config::BASE_ADDR) as usize] as *mut u8
                as *mut VirtioBlkStatus)
                .as_mut()
                .unwrap()
        };

        // manage request.
        virtio_mmio_device.write_u32_impl(VirtIO_MMIO_Offset::QueueNotify as u64, 0x00);

        let interrupt_status =
            virtio_mmio_device.read_u32_impl(VirtIO_MMIO_Offset::InterruptStatus as u64);
        assert_eq!(interrupt_status, 1);
        assert!(virtio_mmio_device.irq_level());
        virtio_mmio_device.write_u32_impl(VirtIO_MMIO_Offset::InterruptAck as u64, 1);
        let interrupt_status =
            virtio_mmio_device.read_u32_impl(VirtIO_MMIO_Offset::InterruptStatus as u64);
        assert_eq!(interrupt_status, 0);
        assert!(!virtio_mmio_device.irq_level());

        // FLUSH uses a two-descriptor chain: request header followed by status.
        *req = VirtioBlkReq::new(VirtioBlkReqType::Flush, 0);
        virt_queue_desc[0].init(
            desc0_buf_addr,
            size_of::<VirtioBlkReq>() as u32,
            VirtQueueDescFlag::VIRTQ_DESC_F_NEXT,
            2,
        );
        desc_status.status = 0xff;
        avail_ring[1] = 0;
        virtq_avail.idx_atomic_add(1);
        virtio_mmio_device.write_u32_impl(VirtIO_MMIO_Offset::QueueNotify as u64, 0);
        assert_eq!(desc_status.status, VirtIOBlkReqStatus::Ok as u8);
        assert_eq!(virtq_used.get_index(), 2);
        assert_eq!(virtq_used.ring(QUEUE_NUM as u32)[1].get_len(), 1);
        virtq_avail.set_flag(VirtQueueAvailFlag::NoInterrupt);
        assert!(!virtio_mmio_device.irq_level());
        virtq_avail.set_flag(VirtQueueAvailFlag::Default);
        assert!(virtio_mmio_device.irq_level());
        virtio_mmio_device.write_u32_impl(VirtIO_MMIO_Offset::InterruptAck as u64, 1);

        assert_eq!(desc_status.status, VirtIOBlkReqStatus::Ok as u8);
        assert_eq!(desc_buf[0], 0);

        // Check the data written.
        let mut buf: [u8; 512] = [0u8; 512];
        file.seek(std::io::SeekFrom::Start(0)).unwrap();
        file.read(&mut buf).unwrap();
        assert_eq!(buf[93], (93 * 93) as u8);

        // Check file size (device config region).
        let capacity = virtio_mmio_device.read_u32_impl(VirtIO_MMIO_Offset::Config as u64);
        assert_eq!(capacity, 1);
        let capacity_high =
            virtio_mmio_device.read_u32_impl(VirtIO_MMIO_Offset::Config as u64 + 0x04);
        assert_eq!(capacity_high, 0);
        let blk_size = virtio_mmio_device
            .read_u32(VirtIO_MMIO_Offset::Config as u64 + 0x14)
            .unwrap();
        assert_eq!(blk_size, 512);
        let writeback = virtio_mmio_device
            .read_u8(VirtIO_MMIO_Offset::Config as u64 + 0x20)
            .unwrap();
        assert_eq!(writeback, 0);

        virtio_mmio_device.write_u32_impl(VirtIO_MMIO_Offset::Status as u64, 0);
        assert_eq!(
            virtio_mmio_device.read_u32_impl(VirtIO_MMIO_Offset::Status as u64),
            0
        );
        assert_eq!(
            virtio_mmio_device.read_u32_impl(VirtIO_MMIO_Offset::QueueReady as u64),
            0
        );

        let unsupported_features = virtio_reserved_feature::VERSION_1 | (1u128 << 127);
        virtio_mmio_device.set_guest_feature(unsupported_features);
        virtio_mmio_device.write_status(
            VirtIODeviceStatus::ACKNOWLEDGE
                | VirtIODeviceStatus::DRIVER
                | VirtIODeviceStatus::FEATURES_OK,
        );
        let status = virtio_mmio_device.read_u32_impl(VirtIO_MMIO_Offset::Status as u64);
        assert_eq!(status & VirtIODeviceStatus::FEATURES_OK.bits() as u32, 0);

        virtio_mmio_device.write_u32_impl(VirtIO_MMIO_Offset::Status as u64, 0);
        virtio_mmio_device.set_guest_feature(VirtIOBlockFeature::BlockSize.bit());
        virtio_mmio_device.write_status(
            VirtIODeviceStatus::ACKNOWLEDGE
                | VirtIODeviceStatus::DRIVER
                | VirtIODeviceStatus::FEATURES_OK,
        );
        let status = virtio_mmio_device.read_u32_impl(VirtIO_MMIO_Offset::Status as u64);
        assert_eq!(status & VirtIODeviceStatus::FEATURES_OK.bits() as u32, 0);
    }
}
