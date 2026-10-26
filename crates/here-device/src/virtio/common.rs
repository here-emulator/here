use crate::device::virtio::{config::VirtIOFeatureSet, virtio_queue::VirtQueue};

struct VirtIODevice {
    // parent_obj: DeviceState,
    name: *const char,
    status: u8,
    isr: u8,
    queue_sel: u16,
    /**
     * These fields represent a set of VirtIO features at various
     * levels of the stack. @host_features indicates the complete
     * feature set the VirtIO device can offer to the driver.
     * @guest_features indicates which features the VirtIO driver has
     * selected by writing to the feature register. Finally
     * @backend_features represents everything supported by the
     * backend (e.g. vhost) and could potentially be a subset of the
     * total feature set offered by QEMU.
     */
    host_features: VirtIOFeatureSet,
    guest_features: VirtIOFeatureSet,
    backend_features: VirtIOFeatureSet,

    config_len: usize,
    // config: *mut c_void,
    config_vector: u16,
    generation: u32,
    nvectors: i32,
    vq: Box<VirtQueue>,
    // listener: MemoryListener,
    device_id: u16,
    /* @vm_running: current VM running state via virtio_vmstate_change() */
    vm_running: bool,
    broken: bool,            /* device in invalid state, needs reset */
    use_disabled_flag: bool, /* allow use of 'disable' flag when needed */
    disabled: bool,          /* device in temporarily disabled state */
    /**
     * @use_started: true if the @started flag should be used to check the
     * current state of the VirtIO device. Otherwise status bits
     * should be checked for a current status of the device.
     * @use_started is only set via QMP and defaults to true for all
     * modern machines (since 4.1).
     */
    use_started: bool,
    started: bool,
    start_on_kick: bool, /* when virtio 1.0 feature has not been negotiated */
    disable_legacy_check: bool,
    vhost_started: bool,
    // vmstate: *mut VMChangeStateEntry,
    bus_name: *const char,
    device_endian: u8,
    /**
     * @user_guest_notifier_mask: gate usage of ->guest_notifier_mask() callback.
     * This is used to suppress the masking of guest updates for
     * vhost-user devices which are asynchronous by design.
     */
    use_guest_notifier_mask: bool,
    // dma_as: *mut AddressSpace,
    // vector_queues: *mut QLIST_HEAD,
    // next: QTAILQ_ENTRY,
    /**
     * @config_notifier: the event notifier that handles config events
     */
    // config_notifier: EventNotifier,
    device_iotlb_enabled: bool,
}
