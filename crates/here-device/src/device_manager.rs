use std::marker::PhantomData;

use super::DeviceTrait;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ErasedDeviceHandle {
    index: usize,
}

#[derive(Debug)]
pub struct DeviceHandle<D: DeviceTrait> {
    index: usize,
    _phantom: PhantomData<D>,
}

impl<D: DeviceTrait> Clone for DeviceHandle<D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: DeviceTrait> Copy for DeviceHandle<D> {}

impl<D: DeviceTrait> DeviceHandle<D> {
    #[inline]
    fn new(index: usize) -> DeviceHandle<D> {
        Self {
            index,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub(crate) fn erase(self) -> ErasedDeviceHandle {
        ErasedDeviceHandle { index: self.index }
    }
}

pub struct DeviceArenaBuilder {
    devices: Vec<Box<dyn DeviceTrait>>,
}

impl DeviceArenaBuilder {
    pub fn new() -> Self {
        Self { devices: vec![] }
    }

    pub fn register<D: DeviceTrait>(&mut self, device: Box<D>) -> DeviceHandle<D> {
        self.devices.push(device);
        DeviceHandle::new(self.devices.len() - 1)
    }

    pub fn build(self) -> DeviceArena {
        DeviceArena {
            devices: self.devices.into_boxed_slice(),
        }
    }
}

pub struct DeviceArena {
    devices: Box<[Box<dyn DeviceTrait>]>,
}

impl DeviceArena {
    pub fn device<D: DeviceTrait>(&self, handle: DeviceHandle<D>) -> &D {
        let device_ref = &*self.devices[handle.index];
        let ptr = device_ref as *const dyn DeviceTrait as *const () as *const D;
        unsafe { &*ptr }
    }

    pub fn device_mut<D: DeviceTrait>(&mut self, handle: DeviceHandle<D>) -> &mut D {
        let device_ref = &mut *self.devices[handle.index];
        let ptr = device_ref as *mut dyn DeviceTrait as *mut () as *mut D;
        unsafe { &mut *ptr }
    }

    pub(crate) fn erased_device_mut(&mut self, handle: ErasedDeviceHandle) -> &mut dyn DeviceTrait {
        &mut *self.devices[handle.index]
    }
}
