// TODO: Always allow dead code for now, as we want to release the crate while many dead code exists.
#![allow(dead_code)]
// Only Allow dead code in debug mode
// #![cfg_attr(debug_assertions, allow(dead_code))]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(macro_metavar_expr_concat)]
#![feature(likely_unlikely)]
#![feature(unsafe_cell_access)]

#[cfg(all(feature = "native-cli", target_arch = "wasm32"))]
compile_error!("feature 'native-cli' is not supported on wasm32 targets");

#[cfg(all(feature = "web", not(target_arch = "wasm32")))]
compile_error!("feature 'web' requires wasm32 target");

mod clock;
mod cpu;
mod fpu;
mod utils;

#[cfg(feature = "native-cli")]
pub mod gdb;

#[cfg(any(feature = "native-cli", feature = "web"))]
pub mod rvdb;

pub mod board;
pub mod byte_io;
pub mod config;
pub mod device;
pub mod isa;
pub mod load;
pub mod ram;
#[cfg(not(target_arch = "wasm32"))]
pub mod task_spawner;

#[cfg(feature = "web")]
pub mod wasm_api;

pub use config::ram_config;

use crate::device::virtio::virtio_device::VirtIODeviceEnum;
use std::{path::PathBuf, str::FromStr};

#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub dev_type: VirtIODeviceEnum,
    pub path: PathBuf,
}

impl FromStr for DeviceConfig {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.split(':');
        let dev_type = match parts.next() {
            Some("virtio-block") => VirtIODeviceEnum::VirtIOBlock,
            Some("virtio-network") => VirtIODeviceEnum::VirtIONet,
            Some(other) => return Err(format!("Unknown device type: {}", other)),
            None => return Err("Invalid device arguments.".into()),
        };
        let path = PathBuf::from(parts.next().ok_or("Need input a device path.")?);
        Ok(DeviceConfig { dev_type, path })
    }
}
