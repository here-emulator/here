use std::{pin::Pin, thread::JoinHandle};

use tokio::runtime::Builder;
use tokio::sync::watch;

type TaskFuture = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

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

pub struct TaskSpawner {
    tasks: Vec<TaskFuture>,
    cancel_rx: watch::Receiver<bool>,
    cancel_tx: watch::Sender<bool>,
}

pub struct TaskHandle {
    cancel_tx: watch::Sender<bool>,
    thread: Option<JoinHandle<()>>,
}

impl TaskSpawner {
    pub fn new() -> Self {
        let (cancel_tx, cancel_rx) = watch::channel(false);

        Self {
            tasks: Vec::new(),
            cancel_rx,
            cancel_tx,
        }
    }

    pub fn register(&mut self, task: impl DeviceTask) {
        self.tasks.push(Box::pin(task.run(self.cancel_rx.clone())));
    }

    pub fn start(self) -> TaskHandle {
        let cancel_tx = self.cancel_tx;
        let tasks = self.tasks;
        let thread = std::thread::spawn(move || {
            let runtime = Builder::new_current_thread().enable_all().build().unwrap();
            runtime.block_on(async move {
                let tasks: Vec<_> = tasks.into_iter().map(tokio::spawn).collect();
                for task in tasks {
                    let _ = task.await;
                }
            });
        });

        TaskHandle {
            cancel_tx,
            thread: Some(thread),
        }
    }
}

impl Drop for TaskHandle {
    fn drop(&mut self) {
        dbg!("dropping task handle");

        if self.cancel_tx.send(true).is_err() {
            dbg!("all device tasks already stopped");
        }
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                dbg!("device task thread panicked");
            }
        }

        dbg!("dropped task handle");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    use super::*;

    struct Dummy {
        done: Arc<AtomicBool>,
    }

    impl DeviceTask for Dummy {
        async fn run(self, mut cancel: watch::Receiver<bool>) {
            self.done.store(true, Ordering::Relaxed);
            cancel.changed().await.unwrap();
        }
    }

    #[test]
    fn smoke() {
        let mut spawner = TaskSpawner::new();
        let done = Arc::new(AtomicBool::new(false));
        spawner.register(Dummy { done: done.clone() });
        let handle = spawner.start();
        drop(handle);
        assert!(done.load(Ordering::Relaxed));
    }
}
