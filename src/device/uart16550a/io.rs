use tokio::sync::mpsc::{
    Sender, UnboundedReceiver,
    error::{TryRecvError, TrySendError},
};

pub const UART_INPUT_CAPACITY: usize = 1024;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UartIoMode {
    None,
    #[default]
    External,
    #[cfg(feature = "native-cli")]
    Stdio,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum UartIoError {
    #[error("UART host I/O is unavailable in {0:?} mode")]
    Unavailable(UartIoMode),
    #[error("UART input buffer is full after accepting {accepted} bytes")]
    InputFull { accepted: usize },
    #[error("UART input channel is closed after accepting {accepted} bytes")]
    InputClosed { accepted: usize },
}

pub struct UartBytePort {
    pub input_tx: Sender<u8>,
    pub output_rx: UnboundedReceiver<u8>,
}

impl UartBytePort {
    pub fn input_sender(&self) -> Sender<u8> {
        self.input_tx.clone()
    }

    pub fn push_input(&self, bytes: &[u8]) -> Result<(), UartIoError> {
        for (accepted, &byte) in bytes.iter().enumerate() {
            match self.input_tx.try_send(byte) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    return Err(UartIoError::InputFull { accepted });
                }
                Err(TrySendError::Closed(_)) => {
                    return Err(UartIoError::InputClosed { accepted });
                }
            }
        }
        Ok(())
    }

    pub fn take_output(&mut self) -> Vec<u8> {
        let mut output = Vec::new();
        loop {
            match self.output_rx.try_recv() {
                Ok(byte) => output.push(byte),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => return output,
            }
        }
    }

    #[cfg(feature = "native-cli")]
    pub(crate) fn spawn_stdout(self, spawner: &crate::task_spawner::TaskSpawner) {
        let Self { mut output_rx, .. } = self;
        spawner.spawn_task(Box::pin(async move {
            use std::io::Write;

            let mut buffer = [0u8; 4096];
            while let Some(byte) = output_rx.recv().await {
                buffer[0] = byte;
                let mut len = 1;
                while len < buffer.len() {
                    match output_rx.try_recv() {
                        Ok(byte) => {
                            buffer[len] = byte;
                            len += 1;
                        }
                        Err(_) => break,
                    }
                }

                let result = (|| -> std::io::Result<()> {
                    let mut stdout = std::io::stdout().lock();
                    stdout.write_all(&buffer[..len])?;
                    stdout.flush()
                })();

                if let Err(error) = result {
                    log::error!("failed to write UART output to stdout: {error}");
                    return;
                }
            }
        }));
    }
}
