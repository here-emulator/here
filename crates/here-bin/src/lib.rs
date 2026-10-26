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

mod clock;
mod cpu;
mod fpu;
mod utils;

#[cfg(feature = "native-cli")]
pub mod gdb;

#[cfg(any(feature = "native-cli", target_arch = "wasm32"))]
pub mod rvdb;

pub mod board;
pub mod byte_io;
pub mod config;
pub mod device;
pub mod isa;
#[cfg(not(target_arch = "wasm32"))]
mod jit;
pub mod load;
pub mod ram;
#[cfg(not(target_arch = "wasm32"))]
pub mod task_spawner;
pub mod types;

#[cfg(target_arch = "wasm32")]
pub mod wasm_api;

pub use config::ram_config;
