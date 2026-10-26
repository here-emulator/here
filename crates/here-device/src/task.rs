//! Async task contract implemented by devices.
//!
//! The runner that owns the runtime and task lifecycle lives in the board layer.

use std::{future::Future, pin::Pin};

use tokio::sync::watch;

pub type TaskFuture = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

/// Choose [`DeviceTask::run`] when you need cancel control (e.g. flush file writes).
pub trait DeviceTask: Send + Sized + 'static {
    fn run(self, mut cancel: watch::Receiver<bool>) -> impl Future<Output = ()> + Send + 'static {
        async move {
            tokio::select! {
                _ = cancel.changed() => {}
                _ = self.run_simple() => {}
            }
        }
    }

    fn run_simple(self) -> impl Future<Output = ()> + Send + 'static {
        async move {}
    }
}
