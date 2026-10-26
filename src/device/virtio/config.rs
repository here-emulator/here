pub(crate) const VIRT_MAGIC: u32 = 0x74726976; // virt
pub(crate) const VIRT_VERSION: u32 = 0x2;
pub(crate) const VIRT_VENDOR: u32 = 0x4A444E42; /* \'JDNB'/ */
pub(crate) const VIRTQUEUE_MAX_SIZE: u32 = 1024;
pub type VirtIOFeatureSet = u128;
pub(crate) const VIRTIO_MAX_FEATURE_BIT_LEN: usize = 128;

pub(crate) mod virtio_reserved_feature {
    use crate::device::virtio::config::VirtIOFeatureSet;

    pub const NOTIFY_ON_EMPTY: VirtIOFeatureSet = 1 << 24;
    pub const ANY_LAYOUT: VirtIOFeatureSet = 1 << 27;
    pub const INDIRECT_DESC: VirtIOFeatureSet = 1 << 28;
    pub const EVENT_IDX: VirtIOFeatureSet = 1 << 29;
    pub const VERSION_1: VirtIOFeatureSet = 1 << 32;
    pub const ACCESS_PLATFORM: VirtIOFeatureSet = 1 << 33;
    pub const RING_PACKED: VirtIOFeatureSet = 1 << 34;
    pub const IN_ORDER: VirtIOFeatureSet = 1 << 35;
    pub const ORDER_PLATFORM: VirtIOFeatureSet = 1 << 36;
    pub const SR_IOV: VirtIOFeatureSet = 1 << 37;
    pub const NOTIFICATION_DATA: VirtIOFeatureSet = 1 << 38;
    pub const NOTIF_CONFIG_DATA: VirtIOFeatureSet = 1 << 39;
    pub const RING_RESET: VirtIOFeatureSet = 1 << 40;
    pub const ADMIN_VQ: VirtIOFeatureSet = 1 << 41;
}
