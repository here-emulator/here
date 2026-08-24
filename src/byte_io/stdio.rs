#![cfg(feature = "native-cli")]

use std::{
    io::Read,
    sync::{Arc, LazyLock},
    thread,
};

use tokio::{
    runtime::Builder,
    sync::{
        mpsc::{self, error::TrySendError},
        watch,
    },
};

use crate::device::power_manager::{POWER_OFF_CODE, POWER_STATUS};
use std::sync::atomic::Ordering;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StdinHandle(u8);

impl StdinHandle {
    const NONE: StdinHandle = StdinHandle(u8::MAX);
}

pub struct StdinRouter {
    senders: Arc<tokio::sync::Mutex<Vec<mpsc::Sender<u8>>>>,
    target_tx: watch::Sender<StdinHandle>,
}

impl StdinRouter {
    pub fn global() -> &'static Self {
        static INSTANCE: LazyLock<StdinRouter> = LazyLock::new(|| {
            let senders = Arc::new(tokio::sync::Mutex::new(
                Vec::<mpsc::Sender<u8>>::with_capacity(4),
            ));

            let rt = Builder::new_current_thread().enable_all().build().unwrap();
            let (target_tx, mut target_rx) = watch::channel(StdinHandle::NONE);

            let senders_clone = Arc::clone(&senders);

            thread::spawn(move || {
                rt.block_on(async move {
                    let mut host_commands = HostCommandFilter::default();
                    loop {
                        let mut buf = [0u8; 1024];
                        let Ok(n) = std::io::stdin().read(&mut buf) else {
                            return;
                        };
                        if n == 0 {
                            return;
                        }

                        let (forwarded, shutdown_requested) = host_commands.filter(&mut buf[..n]);
                        if shutdown_requested {
                            log::info!("[StdinRouter] Ctrl+A x: requesting exit");
                            POWER_STATUS.store(POWER_OFF_CODE, Ordering::Release);
                        }

                        match forward_bytes(&mut target_rx, &senders_clone, &buf[..forwarded]).await
                        {
                            Err(ForwardError::Closed) => return,
                            _ => continue,
                        }
                    }
                });
            });

            StdinRouter { senders, target_tx }
        });

        &INSTANCE
    }

    pub fn register(&self, channel: mpsc::Sender<u8>) -> StdinHandle {
        let mut senders = self.senders.blocking_lock();

        if senders.len() == u8::MAX as usize {
            panic!("too many channels registered!")
        }

        senders.push(channel);
        StdinHandle((senders.len() - 1) as u8)
    }

    pub fn switch_to(&self, target: StdinHandle) {
        self.target_tx.send_replace(target);
    }

    #[cfg(test)]
    pub(crate) fn current_target(&self) -> StdinHandle {
        *self.target_tx.borrow()
    }
}

#[derive(Default)]
struct HostCommandFilter {
    escape_pending: bool,
}

impl HostCommandFilter {
    fn filter(&mut self, input: &mut [u8]) -> (usize, bool) {
        let mut output_len = 0;
        let mut shutdown_requested = false;

        for index in 0..input.len() {
            let byte = input[index];
            if self.escape_pending {
                self.escape_pending = false;
                if byte == b'x' {
                    shutdown_requested = true;
                } else {
                    input[output_len] = byte;
                    output_len += 1;
                }
            } else if byte == 0x01 {
                self.escape_pending = true;
            } else {
                input[output_len] = byte;
                output_len += 1;
            }
        }

        (output_len, shutdown_requested)
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ForwardError {
    Closed,
    TargetChanged,
    TargetClosed,
}

async fn forward_bytes(
    target_rx: &mut watch::Receiver<StdinHandle>,
    senders: &tokio::sync::Mutex<Vec<tokio::sync::mpsc::Sender<u8>>>,
    buf: &[u8],
) -> Result<(), ForwardError> {
    let target = *target_rx.borrow_and_update();
    if target == StdinHandle::NONE {
        return Ok(());
    }
    let id = target.0 as usize;
    let sender = senders
        .lock()
        .await
        .get(id)
        .cloned()
        .ok_or(ForwardError::TargetClosed)?;
    for &b in buf {
        // use non-blocking send for performance
        match sender.try_send(b) {
            Ok(()) => {
                continue;
            }
            Err(TrySendError::Closed(_)) => {
                if let Err(_) = target_rx.changed().await {
                    return Err(ForwardError::Closed);
                }
                return Err(ForwardError::TargetClosed);
            }
            Err(TrySendError::Full(_)) => {
                // keep going
            }
        }

        tokio::select! {
            biased;  // poll in order

            _ = sender.send(b) => {}

            // discard all not send bytes (as intended)
            rst = target_rx.changed() => {
                match rst {
                    Ok(()) => { return Err(ForwardError::TargetChanged); }
                    Err(_) => { return Err(ForwardError::Closed); }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{DeviceTrait, PlicDevice, uart16550a::Uart16550A};

    fn test_router() -> (StdinRouter, watch::Receiver<StdinHandle>) {
        let (target_tx, target_rx) = watch::channel(StdinHandle::NONE);
        (
            StdinRouter {
                senders: Arc::new(tokio::sync::Mutex::new(Vec::new())),
                target_tx,
            },
            target_rx,
        )
    }

    #[test]
    fn switches_targets_without_moving_old_buffer() {
        let runtime = Builder::new_current_thread().enable_all().build().unwrap();
        let (router, mut target_rx) = test_router();
        let (first_tx, mut first_rx) = mpsc::channel(1);
        let (second_tx, mut second_rx) = mpsc::channel(4);
        let first = router.register(first_tx);
        let second = router.register(second_tx);
        router.switch_to(first);

        runtime.block_on(async {
            let senders = Arc::clone(&router.senders);
            let task =
                tokio::spawn(async move { forward_bytes(&mut target_rx, &senders, b"ab").await });
            while first_rx.len() == 0 {
                tokio::task::yield_now().await;
            }
            router.switch_to(second);

            assert_eq!(task.await.unwrap(), Err(ForwardError::TargetChanged));
            assert_eq!(first_rx.try_recv().unwrap(), b'a');
            assert!(second_rx.try_recv().is_err());
        });
    }

    #[test]
    fn uart_sender_is_registered_directly() {
        let runtime = Builder::new_current_thread().enable_all().build().unwrap();
        let (router, mut target_rx) = test_router();
        let (mut uart, port) = Uart16550A::new();
        let uart_handle = router.register(port.input_sender());
        router.switch_to(uart_handle);
        uart.write_u8(1, 0x01).unwrap();

        runtime
            .block_on(forward_bytes(&mut target_rx, &router.senders, b"uart"))
            .unwrap();

        assert!(uart.irq_level());
        for expected in b"uart" {
            assert_eq!(uart.read_u8(0).unwrap(), *expected);
        }
        assert!(!uart.irq_level());
    }

    #[test]
    fn host_command_is_filtered_before_target_routing() {
        let mut filter = HostCommandFilter::default();
        let mut first = *b"a\x01";

        let (forwarded, shutdown) = filter.filter(&mut first);
        assert!(!shutdown);
        assert_eq!(&first[..forwarded], b"a");

        let mut second = *b"xb\x01z";
        let (forwarded, shutdown) = filter.filter(&mut second);
        assert!(shutdown);
        assert_eq!(&second[..forwarded], b"bz");
    }

    #[test]
    fn host_command_is_recognized_without_a_target() {
        let (router, mut target_rx) = test_router();
        assert_eq!(router.current_target(), StdinHandle::NONE);

        let mut filter = HostCommandFilter::default();
        let mut input = *b"\x01x";
        let (forwarded, shutdown) = filter.filter(&mut input);
        assert!(shutdown);
        assert_eq!(forwarded, 0);

        let runtime = Builder::new_current_thread().enable_all().build().unwrap();
        runtime
            .block_on(forward_bytes(
                &mut target_rx,
                &router.senders,
                &input[..forwarded],
            ))
            .unwrap();
    }
}
