use core::slice;
use std::{
    fs::OpenOptions,
    io::SeekFrom,
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
};

#[cfg(test)]
use std::{
    fs::File as StdFile,
    io::{Read, Seek, Write},
};

use log::{error, info};
use num_enum::TryFromPrimitive;
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt},
    sync::{mpsc, watch},
};

use crate::{
    device::virtio::{
        config::{VIRTQUEUE_MAX_SIZE, VirtIOFeatureSet, virtio_reserved_feature},
        virtio_device::VirtIODeviceTrait,
        virtio_mmio::{VIRTIO_DEVICE_ID_BLOCK, VirtIODeviceStatus},
        virtio_queue::{VirtQueue, VirtQueueCompletion, VirtQueueDesc},
    },
    task_spawner::{DeviceTask, TaskSpawner},
};

pub(super) const SECTOR_SIZE: usize = 512;
const VIRTIO_BLOCK_TASK_CAPACITY: usize = VIRTQUEUE_MAX_SIZE as usize;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[rustfmt::skip]
/// Standard VirtIO block feature bits. A variant is advertised only when the
/// corresponding behavior is implemented and enabled by the device builder.
pub(crate) enum VirtIOBlockFeature {
    SizeMax     = 1 << 1,   // Maximum segment size supported
    SegMax      = 1 << 2,   // Maximum number of segments supported
    Geometry    = 1 << 4,   // Disk geometry available
    Ro          = 1 << 5,   // Device is read-only
    BlockSize   = 1 << 6,   // Block size available
    Flush       = 1 << 9,   // Cache flush command supported
    Topology    = 1 << 10,  // Device exports topology information
    ConfigWce   = 1 << 11,  // Writeback mode available in config
    Multiqueue  = 1 << 12,  // Device supports multiqueue.
    Discard     = 1 << 13,  // Discard command supported
    WriteZeroes = 1 << 14,  // Write zeroes command supported
    Lifetime    = 1 << 15,  // Device supports providing storage lifetime information.
    SecureErase = 1 << 16,  // Secure erase supported
    ZONED       = 1 << 17,  // Zoned block device
}

const VIRTIO_BLK_IMPLEMENTED_FEATURES: VirtIOFeatureSet = VirtIOBlockFeature::BlockSize
    as VirtIOFeatureSet
    | VirtIOBlockFeature::Flush as VirtIOFeatureSet;

impl VirtIOBlockFeature {
    pub(crate) const fn bit(self) -> VirtIOFeatureSet {
        self as VirtIOFeatureSet
    }

    const fn is_implemented(self) -> bool {
        VIRTIO_BLK_IMPLEMENTED_FEATURES & self.bit() != 0
    }
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VirtioBlkGeometry {
    pub(crate) cylinders: u16,
    pub(crate) heads: u8,
    pub(crate) sectors: u8,
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VirtioBlkTopology {
    pub(crate) physical_block_exp: u8,
    pub(crate) alignment_offset: u8,
    pub(crate) min_io_size: u16,
    pub(crate) opt_io_size: u32,
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VirtioBlkZonedCharacteristics {
    pub(crate) zone_sectors: u32,
    pub(crate) max_open_zones: u32,
    pub(crate) max_active_zones: u32,
    pub(crate) max_append_sectors: u32,
    pub(crate) write_granularity: u32,
    pub(crate) model: u8,
    pub(crate) unused2: [u8; 3],
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
#[rustfmt::skip]
pub(crate) struct VirtioBlkConfig {
    pub(crate) capacity: u64,                      // 0x00: Size of the block device (in 512-byte sectors)
    pub(crate) size_max: u32,                      // 0x08: Maximum segment size        (if VIRTIO_BLK_F_SIZE_MAX)
    pub(crate) seg_max: u32,                       // 0x0c: Maximum number of segments  (if VIRTIO_BLK_F_SEG_MAX)
    pub(crate) geometry: VirtioBlkGeometry,        // 0x10: Disk geometry               (if VIRTIO_BLK_F_GEOMETRY)
    pub(crate) blk_size: u32,                      // 0x14: Block size of device        (if VIRTIO_BLK_F_BLK_SIZE)
    pub(crate) topology: VirtioBlkTopology,        // 0x18: Topology information        (if VIRTIO_BLK_F_TOPOLOGY)
    pub(crate) writeback: u8,                      // 0x20: Writeback mode              (if VIRTIO_BLK_F_CONFIG_WCE)
    pub(crate) unused0: u8,                        // 0x21: Padding
    pub(crate) num_queues: u16,                    // 0x22: Number of queues            (if VIRTIO_BLK_F_MQ)
    pub(crate) max_discard_sectors: u32,           // 0x24: Max discard sectors         (if VIRTIO_BLK_F_DISCARD)
    pub(crate) max_discard_seg: u32,               // 0x28: Max discard segments        (if VIRTIO_BLK_F_DISCARD)
    pub(crate) discard_sector_alignment: u32,      // 0x2c: Discard sector alignment    (if VIRTIO_BLK_F_DISCARD)
    pub(crate) max_write_zeroes_sectors: u32,      // 0x30: Max write zeroes sectors    (if VIRTIO_BLK_F_WRITE_ZEROES)
    pub(crate) max_write_zeroes_seg: u32,          // 0x34: Max write zeroes segments   (if VIRTIO_BLK_F_WRITE_ZEROES)
    pub(crate) write_zeroes_may_unmap: u8,         // 0x38: Write zeroes may unmap      (if VIRTIO_BLK_F_WRITE_ZEROES)
    pub(crate) unused1: [u8; 3],                   // 0x39: Padding
    pub(crate) max_secure_erase_sectors: u32,      // 0x3c: Max secure erase sectors        (if VIRTIO_BLK_F_SECURE_ERASE)
    pub(crate) max_secure_erase_seg: u32,          // 0x40: Max secure erase segments       (if VIRTIO_BLK_F_SECURE_ERASE)
    pub(crate) secure_erase_sector_alignment: u32, // 0x44: Secure erase sector alignment   (if VIRTIO_BLK_F_SECURE_ERASE)
    pub(crate) zoned: VirtioBlkZonedCharacteristics, // 0x48: Zoned block characteristics    (if VIRTIO_BLK_F_ZONED)
}

impl VirtioBlkConfig {
    pub(crate) fn new(capacity: u64) -> Self {
        let mut config = Self::default();
        config.capacity = capacity;
        config.blk_size = SECTOR_SIZE as u32;
        config
    }

    fn as_bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self as *const Self as *const u8, size_of::<Self>()) }
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self as *mut Self as *mut u8, size_of::<Self>()) }
    }
}

// ======================================
//      Virtio block request types
// ======================================
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive)]
pub(crate) enum VirtioBlkReqType {
    In = 0,
    Out = 1,
    Flush = 4,
    GetId = 8,
    GetLifetime = 10,
    Discard = 11,
    WriteZeroes = 13,
    SecureErase = 14,
    Unsupported = 0xFFFFFFFF,
}

// Virtio block request header (0x10 bytes)
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub(super) struct VirtioBlkReq {
    request_type: u32, // (VirtioBlkReqStatus)
    reserved: u32,
    sector: u64,
}

#[cfg(test)]
impl VirtioBlkReq {
    pub(super) fn new(request_type: VirtioBlkReqType, sector: u64) -> Self {
        Self {
            request_type: request_type as u32,
            reserved: 0,
            sector,
        }
    }
}

struct VirtIOBlkData {
    data0: u8,
    // data1, data2, ..., dataN
}
impl VirtIOBlkData {
    fn as_mut_slice(&mut self, len: usize) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(&mut self.data0 as *mut u8, len) }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VirtIOBlkReqStatus {
    Ok = 0,
    IoErr = 1,
    Unsupported = 2,
    NotReady = 3,
}

#[repr(transparent)]
/// Guest-provided status byte at the tail of a block request.
pub(super) struct VirtioBlkStatus {
    pub(super) status: AtomicU8,
}
impl VirtioBlkStatus {
    fn write_status(&self, status: VirtIOBlkReqStatus) {
        self.status.swap(status as u8, Ordering::AcqRel);
    }
}

#[derive(Clone, Copy)]
/// One guest-RAM data segment retained for an asynchronous block request.
struct IOTaskRamBuffer {
    addr: usize,
    len: usize,
}

/// Guest-visible state required to publish a completed block request.
struct VirtIOBlockCompletion {
    status_addr: usize,
    queue_completion: VirtQueueCompletion,
    isr: Arc<AtomicU8>,
}

impl VirtIOBlockCompletion {
    fn complete(self, status: VirtIOBlkReqStatus, used_len: u32) {
        // The status byte and used-ring entry are guest-owned memory. The
        // guest must not reuse these buffers until the used index is published.
        unsafe { &*(self.status_addr as *const VirtioBlkStatus) }.write_status(status);
        self.queue_completion.complete(used_len);
        self.isr.fetch_or(1, Ordering::AcqRel);
    }
}

/// Host I/O parameters for one read or write block request.
struct VirtIOBlockIoTask {
    offset: u64,
    buffers: Vec<IOTaskRamBuffer>,
    completion: VirtIOBlockCompletion,
}

/// Work item processed by the asynchronous VirtIO block worker.
enum VirtIOBlockAsyncTask {
    Read(VirtIOBlockIoTask),
    Write(VirtIOBlockIoTask),
    Flush(VirtIOBlockCompletion),
    EarlyErr(VirtIOBlockCompletion), // Early Error(like offset out of bound)
    Unsupported(VirtIOBlockCompletion),
    /// Acknowledge after all tasks submitted before this marker are complete.
    DrainBarrier(std::sync::mpsc::SyncSender<()>),
}

/// Owns the backing file and executes queued block requests on the task runtime.
pub(super) struct VirtIOBlockAsyncWorker {
    file: File,
    receiver: mpsc::Receiver<VirtIOBlockAsyncTask>,
}

impl DeviceTask for VirtIOBlockAsyncWorker {
    async fn run(mut self, mut cancel: watch::Receiver<bool>) {
        loop {
            let task = tokio::select! {
                _ = cancel.changed() => return,
                task = self.receiver.recv() => match task {
                    Some(task) => task,
                    None => return,
                },
            };

            self.run_task(task).await;
            if *cancel.borrow() {
                return;
            }
        }
    }
}

impl VirtIOBlockAsyncWorker {
    fn new(file: File, receiver: mpsc::Receiver<VirtIOBlockAsyncTask>) -> Self {
        Self { file, receiver }
    }

    async fn run_task(&mut self, task: VirtIOBlockAsyncTask) {
        // Synchronize guest writes published before QueueNotify with this worker.
        std::sync::atomic::fence(Ordering::Acquire);

        match task {
            VirtIOBlockAsyncTask::Read(task) => self.read(task).await,
            VirtIOBlockAsyncTask::Write(task) => self.write(task).await,
            VirtIOBlockAsyncTask::Flush(completion) => {
                let status = if self.file.sync_data().await.is_ok() {
                    VirtIOBlkReqStatus::Ok
                } else {
                    VirtIOBlkReqStatus::IoErr
                };
                completion.complete(status, size_of::<VirtioBlkStatus>() as u32);
            }
            VirtIOBlockAsyncTask::EarlyErr(completion) => {
                completion.complete(
                    VirtIOBlkReqStatus::IoErr,
                    size_of::<VirtioBlkStatus>() as u32,
                );
            }
            VirtIOBlockAsyncTask::Unsupported(completion) => {
                completion.complete(
                    VirtIOBlkReqStatus::Unsupported,
                    size_of::<VirtioBlkStatus>() as u32,
                );
            }
            VirtIOBlockAsyncTask::DrainBarrier(ack) => {
                // The worker is single-threaded, so reaching this marker means
                // every earlier request has finished publishing its completion.
                let _ = ack.send(());
            }
        }
    }

    async fn read(&mut self, task: VirtIOBlockIoTask) {
        let mut offset = task.offset;
        let mut total_transferred = 0;
        let mut status = VirtIOBlkReqStatus::Ok;

        for buffer in &task.buffers {
            let data = unsafe { slice::from_raw_parts_mut(buffer.addr as *mut u8, buffer.len) };
            if self.file.seek(SeekFrom::Start(offset)).await.is_err() {
                status = VirtIOBlkReqStatus::IoErr;
                break;
            }
            let Ok(len) = self.file.read(data).await else {
                status = VirtIOBlkReqStatus::IoErr;
                break;
            };

            total_transferred += len as u32;
            offset += len as u64;
            if len != data.len() {
                status = VirtIOBlkReqStatus::IoErr;
                break;
            }
        }

        task.completion.complete(
            status,
            total_transferred + size_of::<VirtioBlkStatus>() as u32,
        );
    }

    async fn write(&mut self, task: VirtIOBlockIoTask) {
        let mut offset = task.offset;
        let mut status = VirtIOBlkReqStatus::Ok;

        for buffer_info in &task.buffers {
            let buffer =
                unsafe { slice::from_raw_parts(buffer_info.addr as *const u8, buffer_info.len) };
            if self.file.seek(SeekFrom::Start(offset)).await.is_err()
                || self.file.write_all(buffer).await.is_err()
            {
                status = VirtIOBlkReqStatus::IoErr;
                break;
            }
            offset += buffer.len() as u64;
        }

        task.completion
            .complete(status, size_of::<VirtioBlkStatus>() as u32);
    }

    #[cfg(test)]
    pub(super) fn process_one(&mut self) -> bool {
        match self.receiver.try_recv() {
            Ok(task) => {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(self.run_task(task));
                true
            }
            Err(mpsc::error::TryRecvError::Empty | mpsc::error::TryRecvError::Disconnected) => {
                false
            }
        }
    }
}

// ======================================
//          Virtio Block Device
// ======================================
pub(crate) struct VirtIOBlkDevice {
    pub(crate) name: &'static str,
    pub(crate) status: u8,
    pub(crate) isr: Arc<AtomicU8>,

    host_feature: VirtIOFeatureSet,
    guest_feature: VirtIOFeatureSet,

    pub(crate) generation: u32,
    ram_base_raw: usize,

    sender: mpsc::Sender<VirtIOBlockAsyncTask>,

    queue: VirtQueue,
    pub(super) config_region: VirtioBlkConfig,
}

impl VirtIOBlkDevice {
    fn new(
        name: &'static str,
        ram_base_raw: *mut u8,
        capacity: u64,
        sender: mpsc::Sender<VirtIOBlockAsyncTask>,
    ) -> Self {
        info!("build virtio block device.");

        Self {
            name,
            status: 0,

            isr: Arc::new(AtomicU8::new(0)),

            host_feature: virtio_reserved_feature::VERSION_1,
            guest_feature: 0,

            generation: 0,
            ram_base_raw: ram_base_raw as usize,

            sender,

            queue: VirtQueue::new(ram_base_raw, 0), // will be set later
            config_region: VirtioBlkConfig::new(capacity),
        }
    }

    pub fn add_host_feature(mut self, new_feature: VirtIOBlockFeature) -> Self {
        assert!(
            new_feature.is_implemented(),
            "VirtIO block feature {new_feature:?} is not implemented"
        );
        self.host_feature |= new_feature.bit();
        self
    }

    #[cfg(test)]
    fn write_blk(file: &mut StdFile, buf: &[u8], offset: u64) -> u32 {
        file.seek(std::io::SeekFrom::Start(offset)).unwrap();
        match file.write_all(buf) {
            Ok(_) => buf.len() as u32,
            Err(_) => 0,
        }
    }

    #[cfg(test)]
    fn read_blk(file: &mut StdFile, buf: &mut [u8], offset: u64) -> u32 {
        file.seek(std::io::SeekFrom::Start(offset)).unwrap();
        match file.read(buf) {
            Ok(len) => len as u32,
            Err(mes) => panic!("{}", mes),
        }
    }

    fn manage_request_header(ram_base_raw: usize, desc: &VirtQueueDesc) -> (VirtioBlkReqType, u64) {
        const BAD_REQ: (VirtioBlkReqType, u64) = (VirtioBlkReqType::Unsupported, 0);
        if desc.len < size_of::<VirtioBlkReq>() as u32 {
            return BAD_REQ;
        }
        let req = unsafe {
            desc.get_request_package::<VirtioBlkReq>(ram_base_raw)
                .as_ref()
                .unwrap()
        };

        VirtioBlkReqType::try_from(req.request_type)
            .map_or(BAD_REQ, |req_type| (req_type, req.sector))
    }

    fn request_offset_in_bounds(&self, sector: u64, buffers: &[IOTaskRamBuffer]) -> Option<u64> {
        let offset = sector.checked_mul(SECTOR_SIZE as u64)?;
        let data_len = buffers.iter().try_fold(0u64, |total, buffer| {
            total.checked_add(u64::try_from(buffer.len).ok()?)
        })?;
        let capacity = self
            .config_region
            .capacity
            .checked_mul(SECTOR_SIZE as u64)?;

        if data_len > u32::MAX as u64 - size_of::<VirtioBlkStatus>() as u64
            || offset.checked_add(data_len)? > capacity
        {
            return None;
        }

        Some(offset)
    }
}

impl VirtIODeviceTrait for VirtIOBlkDevice {
    fn get_device_id(&self) -> u16 {
        info!("get block device id.");
        VIRTIO_DEVICE_ID_BLOCK
    }

    fn status(&mut self) -> &mut u8 {
        &mut self.status
    }

    fn get_generation(&self) -> u32 {
        self.generation
    }

    fn reset(&mut self) {
        info!("reset virtio block device.");

        // VirtIO requires all outstanding requests to be finished or discarded
        // before the device reports status 0. The worker consumes requests in
        // FIFO order, so a barrier drains all requests submitted before reset.
        // If the worker stops before consuming the marker, its acknowledgement
        // sender is dropped with the marker and recv() returns an error; no task
        // remains that can access guest RAM.
        let (drain_tx, drain_rx) = std::sync::mpsc::sync_channel(0);
        if self
            .sender
            .blocking_send(VirtIOBlockAsyncTask::DrainBarrier(drain_tx))
            .is_ok()
        {
            let _ = drain_rx.recv();
        }

        self.status = 0;
        self.guest_feature = 0;
        self.queue.reset();
        self.isr.store(0, Ordering::Release);
    }

    fn isr(&self) -> &AtomicU8 {
        &self.isr
    }

    fn irq_level(&mut self) -> bool {
        self.isr.load(Ordering::Acquire) != 0 && self.queue.interrupts_enabled()
    }

    fn get_host_feature(&self) -> VirtIOFeatureSet {
        self.host_feature
    }

    fn set_feature(&mut self, feature: VirtIOFeatureSet) {
        info!("set virtio block feature.");
        if feature & virtio_reserved_feature::VERSION_1 == 0
            || self.host_feature & feature != feature
        {
            self.status &= !VirtIODeviceStatus::FEATURES_OK.bits();
        } else {
            self.guest_feature = feature;
        }
    }

    fn set_queue_num(&mut self, num: u32) {
        self.queue.set_queue_num(num);
    }

    fn queue_select(&self, _idx: u32) {
        // ONLY ONE QUEUE.
    }

    fn set_desc(&mut self, addr: u64) {
        self.queue.set_desc(addr);
    }

    fn set_avail(&mut self, addr: u64) {
        self.queue.set_avail(addr);
    }

    fn set_used(&mut self, addr: u64) {
        self.queue.set_used(addr);
    }

    /// Submit one complete VirtIO block I/O request to the host worker.
    ///
    /// A request descriptor chain contains at least three parts:
    /// the request header (`desc.addr` points to [`VirtioBlkReq`]), which
    /// describes the operation; one or more request-body descriptors whose
    /// `desc.addr` values point into guest RAM shared with the host, where the
    /// data is read from or written to according to the header; and a request
    /// tail whose descriptor returns the final [`VirtIOBlkReqStatus`]. The
    /// used-ring entry, tail status, and interrupt status are published by the
    /// worker only after the host I/O completes.
    fn manage_one_request(&mut self) -> bool {
        let Ok(permit) = self.sender.try_reserve() else {
            return false;
        };

        info!("[virtio-block]: submit a request.");
        let mut req_type = VirtioBlkReqType::Unsupported;
        let mut sector = 0;
        let mut buffers = Vec::new();
        let mut status_addr = None;
        let ram_base_raw = self.ram_base_raw;
        let Some(used) =
            self.queue
                .take_one_request(|desc: &VirtQueueDesc, idx: usize| match idx {
                    // Request header: decode the operation and starting sector.
                    0 => {
                        (req_type, sector) = Self::manage_request_header(ram_base_raw, desc);
                        info!("[virtio-block]: req_type = {:?}.", req_type);
                    }

                    // Request body: retain the shared RAM buffer for the worker.
                    _ if desc.has_next() => {
                        buffers.push(IOTaskRamBuffer {
                            addr: desc.get_request_package::<u8>(ram_base_raw) as usize,
                            len: desc.len as usize,
                        });
                    }

                    // Request tail: retain the final status byte for the worker.
                    _ => {
                        if desc.len < size_of::<VirtioBlkStatus>() as u32 {
                            return;
                        }
                        status_addr = Some(
                            desc.get_request_package::<VirtioBlkStatus>(ram_base_raw) as usize,
                        );
                    }
                })
        else {
            return false;
        };

        let Some(status_addr) = status_addr else {
            used.complete(0);
            return true;
        };
        let completion = VirtIOBlockCompletion {
            status_addr,
            queue_completion: used,
            isr: self.isr.clone(),
        };
        let data_offset = self.request_offset_in_bounds(sector, &buffers);
        let task = match req_type {
            VirtioBlkReqType::In => match data_offset {
                Some(offset) => VirtIOBlockAsyncTask::Read(VirtIOBlockIoTask {
                    offset,
                    buffers,
                    completion,
                }),
                None => VirtIOBlockAsyncTask::EarlyErr(completion),
            },
            VirtioBlkReqType::Out => match data_offset {
                Some(offset) => VirtIOBlockAsyncTask::Write(VirtIOBlockIoTask {
                    offset,
                    buffers,
                    completion,
                }),
                None => VirtIOBlockAsyncTask::EarlyErr(completion),
            },
            VirtioBlkReqType::Flush => VirtIOBlockAsyncTask::Flush(completion),
            _ => VirtIOBlockAsyncTask::Unsupported(completion),
        };
        permit.send(task);
        true
    }

    fn notify(&mut self, _idx: u32) {
        loop {
            if !self.manage_one_request() {
                break;
            }
        }
    }

    fn queue_ready(&self) -> bool {
        self.queue.ready()
    }

    fn get_num_of_queue(&self) -> u32 {
        1
    }

    fn read_config(&mut self, offset: u64, len: u32) -> u64 {
        let offset = offset as usize;
        let len = len as usize;
        let bytes = self.config_region.as_bytes();
        if offset.checked_add(len).is_none_or(|end| end > bytes.len()) {
            error!("virtio-blk config read out of range: offset={offset:#x}, len={len}");
            return 0;
        }

        let mut value = 0u64;
        for (idx, byte) in bytes[offset..offset + len].iter().enumerate() {
            value |= (*byte as u64) << (idx * u8::BITS as usize);
        }
        value
    }

    fn write_config(&mut self, offset: u64, len: u32, data: u64) {
        let offset = offset as usize;
        let len = len as usize;
        let bytes = self.config_region.as_bytes_mut();
        if offset.checked_add(len).is_none_or(|end| end > bytes.len()) {
            error!("virtio-blk config write out of range: offset={offset:#x}, len={len}");
            return;
        }

        for (idx, byte) in bytes[offset..offset + len].iter_mut().enumerate() {
            *byte = (data >> (idx * u8::BITS as usize)) as u8;
        }
    }
}

#[cfg(test)]
impl VirtIOBlkDevice {
    pub(crate) fn queue(&mut self) -> &mut VirtQueue {
        &mut self.queue
    }
}

pub struct VirtIOBlkDeviceBuilder {
    device: VirtIOBlkDevice,
    receiver: mpsc::Receiver<VirtIOBlockAsyncTask>,
    file: File,
}

impl VirtIOBlkDeviceBuilder {
    pub fn new(ram_base_raw: *mut u8, file: String) -> Self {
        let (sender, receiver) = mpsc::channel(VIRTIO_BLOCK_TASK_CAPACITY);
        let (backing_file, size) = Self::open_backing_file(&file);
        if !size.is_multiple_of(SECTOR_SIZE as u64) {
            panic!(
                "VirtIO block backing file \"{file}\" has size {size} bytes; size must be a multiple of {SECTOR_SIZE} bytes"
            );
        }

        Self {
            device: VirtIOBlkDevice::new(
                "Unnamed VirtIO Block Device",
                ram_base_raw,
                size / SECTOR_SIZE as u64,
                sender,
            ),
            receiver,
            file: backing_file,
        }
    }

    fn open_backing_file(path: &str) -> (File, u64) {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .append(false)
            .create(false)
            .open(path)
            .unwrap_or_else(|_| panic!("Can not find file: {path}."));
        let size = file.metadata().unwrap().len();
        (File::from_std(file), size)
    }

    pub fn name(mut self, name: &'static str) -> Self {
        self.device.name = name;
        self
    }

    pub fn host_feature(mut self, feature: VirtIOBlockFeature) -> Self {
        assert!(
            feature.is_implemented(),
            "VirtIO block feature {feature:?} is not implemented"
        );
        self.device.host_feature |= feature.bit();
        self
    }

    pub fn generation(mut self, generation: u32) -> Self {
        self.device.generation = generation;
        self
    }

    pub fn get_and_spawner_task(self, task_spawner: &mut TaskSpawner) -> VirtIOBlkDevice {
        let Self {
            device,
            receiver,
            file,
        } = self;
        task_spawner.register(VirtIOBlockAsyncWorker::new(file, receiver));
        device
    }

    #[cfg(test)]
    pub(crate) fn get(self) -> VirtIOBlkDevice {
        self.get_with_worker().0
    }

    #[cfg(test)]
    pub(super) fn get_with_worker(self) -> (VirtIOBlkDevice, VirtIOBlockAsyncWorker) {
        let Self {
            device,
            receiver,
            file,
        } = self;
        (device, VirtIOBlockAsyncWorker::new(file, receiver))
    }
}

#[cfg(test)]
pub fn init_block_file<'a, F>(path: &str, blk_num: u64, mut f: F) -> StdFile
where
    F: FnMut(usize) -> &'a [u8],
{
    use std::{fs::create_dir_all, path::Path};
    let parent_dir = Path::new(path).parent().unwrap();
    create_dir_all(parent_dir).unwrap();

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .unwrap();
    // let write_buf: [u8; SECTOR_SIZE] = [0; SECTOR_SIZE];
    for i in 0..blk_num {
        let write_buf = f(i as usize);
        assert_eq!(write_buf.len(), SECTOR_SIZE);
        file.write_all(write_buf).unwrap();
    }

    file
}

#[cfg(test)]
mod test {
    use std::{
        thread,
        time::{Duration, Instant},
    };

    use crate::{
        device::virtio::virtio_queue::{
            VirtQueueAvail, VirtQueueAvailFlag, VirtQueueDescFlag, VirtQueueUsed, VirtQueueUsedFlag,
        },
        ram::Ram,
        ram_config,
    };

    use super::*;
    const QUEUE_NUM: usize = 8;
    const DESC_NUM: usize = QUEUE_NUM * 3; // each request need

    fn wait_for_used_index(used_ring: &VirtQueueUsed, expected: u16) {
        let deadline = Instant::now() + Duration::from_secs(1);
        while used_ring.get_index() != expected {
            assert!(Instant::now() < deadline, "virtio block worker timed out");
            thread::yield_now();
        }
    }

    #[test]
    #[should_panic(expected = "size must be a multiple of 512 bytes")]
    fn rejects_unaligned_backing_file_size() {
        use std::{fs::create_dir_all, path::Path};

        let file_name = "./tmp/test_unaligned_virtio_blk.img";
        create_dir_all(Path::new(file_name).parent().unwrap()).unwrap();
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(file_name)
            .unwrap();
        file.set_len(SECTOR_SIZE as u64 + 1).unwrap();
        drop(file);

        let mut ram = Ram::new();
        let ram_base = &mut ram[0] as *mut u8;
        let _ = VirtIOBlkDeviceBuilder::new(ram_base, file_name.to_owned()).get();
    }

    #[test]
    fn implemented_feature_set_matches_block_handlers() {
        assert!(VirtIOBlockFeature::BlockSize.is_implemented());
        assert!(VirtIOBlockFeature::Flush.is_implemented());
        assert!(!VirtIOBlockFeature::Discard.is_implemented());
        assert!(!VirtIOBlockFeature::WriteZeroes.is_implemented());
    }

    #[test]
    fn test_file_read_write() {
        // let mut file = OpenOptions::new()
        //     .read(true)
        //     .write(true)
        //     .create(true)
        //     .truncate(true)
        //     .open("./tmp/test_file_read_write.txt")
        //     .unwrap();
        let write_buf: [u8; SECTOR_SIZE] = [0xAB; SECTOR_SIZE];
        let offset = 0;
        let mut file = init_block_file("./tmp/test_file_read_write.txt", 1, |_| &write_buf);

        // 测试写入
        let write_len = VirtIOBlkDevice::write_blk(&mut file, &write_buf, offset);
        assert_eq!(write_len, SECTOR_SIZE as u32);

        let mut file_copy = file.try_clone().unwrap();
        // 测试读取
        let mut read_buf: [u8; SECTOR_SIZE] = [0u8; SECTOR_SIZE];
        let read_len = VirtIOBlkDevice::read_blk(&mut file_copy, &mut read_buf, offset);
        assert_eq!(read_len, SECTOR_SIZE as u32);
        assert_eq!(read_buf, write_buf);
    }

    #[test]
    fn test_blk_read() {
        let mut buf: [u8; SECTOR_SIZE] = [0u8; SECTOR_SIZE];
        buf[0xff] = 0x55;
        let file_name = String::from("./tmp/test_blk_read.txt");
        let _ = init_block_file(&file_name, 1, |_| &buf);

        let mut ram = Ram::new();
        let ram_base = &mut ram[0] as *mut u8;
        let mut spawner = TaskSpawner::new();
        let mut virt_device =
            VirtIOBlkDeviceBuilder::new(ram_base, file_name).get_and_spawner_task(&mut spawner);
        let _task_handle = spawner.start();
        virt_device.set_queue_num(QUEUE_NUM as u32);

        let virtq_desc_base = 0x8000_2000 as u64;
        let virtq_avail_base = 0x8000_2100 + ((QUEUE_NUM + 2) * size_of::<u16>()) as u64;
        let virtq_used_base = 0x8000_2200 + (QUEUE_NUM * size_of::<VirtQueueUsed>() + 4) as u64;
        virt_device.set_avail(virtq_avail_base);
        virt_device.set_desc(virtq_desc_base);
        virt_device.set_used(virtq_used_base);

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
        let avail_ring = VirtQueueAvail::mut_ring(virtq_avail as *mut _ as u64, QUEUE_NUM as u32);

        // Used Ring.
        let virtq_used = &mut ram[(virtq_used_base - ram_config::BASE_ADDR) as usize] as *mut u8
            as *mut VirtQueueUsed;
        let virtq_used = unsafe { virtq_used.as_mut().unwrap() };
        virtq_used.init(VirtQueueUsedFlag::Default);
        let _used_ring = virtq_used.ring(QUEUE_NUM as u32);

        // Write Available Ring.
        avail_ring[0] = 0;
        virtq_avail.idx_atomic_add(1);

        // header
        let desc0 = &mut virt_queue_desc[0];
        let desc0_buf_addr = 0x8000_2300;
        desc0.init(
            0x8000_2300,
            size_of::<VirtioBlkReq>() as u32,
            VirtQueueDescFlag::VIRTQ_DESC_F_NEXT,
            1,
        );
        let req = &mut ram[(desc0_buf_addr - ram_config::BASE_ADDR) as usize] as *mut u8
            as *mut VirtioBlkReq;
        let req = unsafe { req.as_mut().unwrap() };
        req.request_type = VirtioBlkReqType::In as u32;
        req.reserved = 0;
        req.sector = 0;

        let desc1 = &mut virt_queue_desc[1];
        let desc1_buf_addr = 0x8000_2400;
        desc1.init(0x8000_2400, 0x200, VirtQueueDescFlag::VIRTQ_DESC_F_NEXT, 2);
        let desc_buf = unsafe {
            slice::from_raw_parts_mut(
                &mut ram[(desc1_buf_addr - ram_config::BASE_ADDR) as usize] as *mut u8,
                0x200,
            )
        };

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
        let t = virt_device.manage_one_request();
        assert_eq!(t, true);
        wait_for_used_index(virtq_used, 1);

        assert_eq!(
            desc_status.status.load(Ordering::Acquire),
            VirtIOBlkReqStatus::Ok as u8
        );
        assert_eq!(desc_buf[0], 0);

        let used_ring = virt_device.queue.get_used_ring();
        let used_index = used_ring.get_index();
        assert_eq!(used_index, 1);
        // used_ring.index_add(1);

        let used_elem = used_ring.ring(QUEUE_NUM as u32)[0];
        assert_eq!(used_elem.get_len(), 0x201);
        assert_eq!(used_elem.get_id(), 0);
    }

    #[test]
    fn test_blk_write() {
        // init file.
        let mut buf: [u8; SECTOR_SIZE] = [0u8; SECTOR_SIZE];
        buf[0xff] = 0x55;
        let file_name = String::from("./tmp/test_blk_write.txt");
        let mut file = init_block_file(file_name.as_str(), 1, |_| &buf);

        let mut ram = Ram::new();
        let ram_base = &mut ram[0] as *mut u8;
        let (mut virt_device, mut worker) =
            VirtIOBlkDeviceBuilder::new(ram_base, file_name).get_with_worker();
        virt_device.set_queue_num(QUEUE_NUM as u32);

        let virtq_desc_base = 0x8000_2000 as u64;
        let virtq_avail_base = 0x8000_2100 + ((QUEUE_NUM + 2) * size_of::<u16>()) as u64;
        let virtq_used_base = 0x8000_2200 + (QUEUE_NUM * size_of::<VirtQueueUsed>() + 4) as u64;
        virt_device.set_avail(virtq_avail_base);
        virt_device.set_desc(virtq_desc_base);
        virt_device.set_used(virtq_used_base);

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
        let avail_ring = VirtQueueAvail::mut_ring(virtq_avail as *mut _ as u64, QUEUE_NUM as u32);

        // Used Ring.
        let virtq_used = &mut ram[(virtq_used_base - ram_config::BASE_ADDR) as usize] as *mut u8
            as *mut VirtQueueUsed;
        let virtq_used = unsafe { virtq_used.as_mut().unwrap() };
        virtq_used.init(VirtQueueUsedFlag::Default);
        let _used_ring = virtq_used.ring(QUEUE_NUM as u32);

        // Write Available Ring.
        avail_ring[0] = 0;
        virtq_avail.idx_atomic_add(1);

        // header
        let desc0 = &mut virt_queue_desc[0];
        let desc0_buf_addr = 0x8000_2300;
        desc0.init(
            0x8000_2300,
            size_of::<VirtioBlkReq>() as u32,
            VirtQueueDescFlag::VIRTQ_DESC_F_NEXT,
            1,
        );
        let req = &mut ram[(desc0_buf_addr - ram_config::BASE_ADDR) as usize] as *mut u8
            as *mut VirtioBlkReq;
        let req = unsafe { req.as_mut().unwrap() };
        req.request_type = VirtioBlkReqType::Out as u32;
        req.reserved = 0;
        req.sector = 0;

        let desc1 = &mut virt_queue_desc[1];
        let desc1_buf_addr = 0x8000_2400;
        desc1.init(0x8000_2400, 0x100, VirtQueueDescFlag::VIRTQ_DESC_F_NEXT, 2);
        let desc_buf = unsafe {
            slice::from_raw_parts_mut(
                &mut ram[(desc1_buf_addr - ram_config::BASE_ADDR) as usize] as *mut u8,
                0x100,
            )
        };
        for i in 0..0x100 {
            desc_buf[i] = (i * i) as u8;
        }

        let desc2 = &mut virt_queue_desc[2];
        let desc2_buf_addr = 0x8000_2500;
        desc2.init(
            desc2_buf_addr,
            0x100,
            VirtQueueDescFlag::VIRTQ_DESC_F_NEXT,
            3,
        );
        let desc_buf2 = unsafe {
            slice::from_raw_parts_mut(
                &mut ram[(desc2_buf_addr - ram_config::BASE_ADDR) as usize] as *mut u8,
                0x100,
            )
        };
        for (idx, byte) in desc_buf2.iter_mut().enumerate() {
            *byte = (idx * 3) as u8;
        }

        let desc3 = &mut virt_queue_desc[3];
        let desc3_buf_addr = 0x8000_2310;
        desc3.init(
            desc3_buf_addr,
            size_of::<VirtioBlkStatus>() as u32, // 1 byte
            VirtQueueDescFlag::empty(),
            0,
        );
        let desc_status = unsafe {
            (&mut ram[(desc3_buf_addr - ram_config::BASE_ADDR) as usize] as *mut u8
                as *mut VirtioBlkStatus)
                .as_mut()
                .unwrap()
        };

        // manage request.
        let t = virt_device.manage_one_request();
        assert_eq!(t, true);
        assert!(worker.process_one());

        assert_eq!(
            desc_status.status.load(Ordering::Acquire),
            VirtIOBlkReqStatus::Ok as u8
        );
        assert_eq!(desc_buf[0], 0);

        let used_ring = virt_device.queue.get_used_ring();
        let used_index = used_ring.get_index();
        assert_eq!(used_index, 1);
        // used_ring.index_add(1);

        let used_elem = used_ring.ring(QUEUE_NUM as u32)[0];
        assert_eq!(used_elem.get_len(), 1);
        assert_eq!(used_elem.get_id(), 0);

        let mut buf: [u8; SECTOR_SIZE] = [0u8; SECTOR_SIZE];
        file.seek(std::io::SeekFrom::Start(0)).unwrap();
        file.read(&mut buf).unwrap();
        assert_eq!(buf[93], (93 * 93) as u8);
        assert_eq!(buf[SECTOR_SIZE / 2 + 93], (93 * 3) as u8);

        // An OUT request beyond the advertised capacity must not extend the backing file.
        req.sector = 1;
        desc_status.status.store(0xff, Ordering::Relaxed);
        avail_ring[1] = 0;
        virtq_avail.idx_atomic_add(1);
        assert!(virt_device.manage_one_request());
        assert!(worker.process_one());
        assert_eq!(
            desc_status.status.load(Ordering::Acquire),
            VirtIOBlkReqStatus::IoErr as u8
        );
        assert_eq!(virt_device.queue.get_used_ring().get_index(), 2);
        assert_eq!(file.metadata().unwrap().len(), SECTOR_SIZE as u64);

        // The checked sector-to-byte conversion rejects integer overflow too.
        req.sector = u64::MAX;
        desc_status.status.store(0xff, Ordering::Relaxed);
        avail_ring[2] = 0;
        virtq_avail.idx_atomic_add(1);
        assert!(virt_device.manage_one_request());
        assert!(worker.process_one());
        assert_eq!(
            desc_status.status.load(Ordering::Acquire),
            VirtIOBlkReqStatus::IoErr as u8
        );
        assert_eq!(virt_device.queue.get_used_ring().get_index(), 3);
        assert_eq!(file.metadata().unwrap().len(), SECTOR_SIZE as u64);
    }
}
