use std::io::Write as StdWrite;

use embedded_io::{ErrorKind, ErrorType, Read, Write};
use noline::{
    builder::EditorBuilder, error::NolineError, history::UnboundedHistory,
    line_buffer::UnboundedBuffer, sync_editor::Editor,
};
use tokio::sync::mpsc;

use crate::{
    board::Board,
    byte_io::{StdinHandle, StdinRouter},
};

use super::{
    RvdbCommand, RvdbSession, format_clap_error, is_clap_display, session::RvdbCommandResponse,
};

const REPL_INPUT_CAPACITY: usize = 1024;

struct SyncReplIo {
    input: mpsc::Receiver<u8>,
}

impl ErrorType for SyncReplIo {
    type Error = ErrorKind;
}

impl Read for SyncReplIo {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        if buf.is_empty() {
            return Ok(0);
        }

        let Some(first) = self.input.blocking_recv() else {
            return Ok(0);
        };
        buf[0] = first;
        let mut count = 1;
        while count < buf.len() {
            match self.input.try_recv() {
                Ok(byte) => {
                    buf[count] = byte;
                    count += 1;
                }
                Err(_) => break,
            }
        }
        Ok(count)
    }
}

impl Write for SyncReplIo {
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        std::io::stdout().write(buf).map_err(|_| ErrorKind::Other)
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        std::io::stdout().flush().map_err(|_| ErrorKind::Other)
    }
}

struct RestoreStdinTarget {
    target: StdinHandle,
}

fn with_stdin_target<T>(target: StdinHandle, restore: StdinHandle, f: impl FnOnce() -> T) -> T {
    StdinRouter::global().switch_to(target);
    let _restore = RestoreStdinTarget { target: restore };
    f()
}

fn resolve_line(line: &str, last_line: &mut String) -> Option<String> {
    let line = line.trim();
    if line.is_empty() {
        (!last_line.is_empty()).then(|| last_line.clone())
    } else {
        *last_line = line.to_owned();
        Some(line.to_owned())
    }
}

/// linux won't handle LF to CRLF on raw mode
fn write_crlf(bytes: &[u8]) -> std::io::Result<()> {
    let mut stdout = std::io::stdout().lock();
    let mut last = 0;
    for (curr, &byte) in bytes.iter().enumerate() {
        if byte != b'\n' {
            continue;
        }

        stdout.write_all(&bytes[last..curr])?;
        if curr == 0 || bytes[curr - 1] != b'\r' {
            stdout.write_all(b"\r")?;
        }
        stdout.write_all(b"\n")?;
        last = curr + 1;
    }
    stdout.write_all(&bytes[last..])?;
    stdout.flush()
}

impl Drop for RestoreStdinTarget {
    fn drop(&mut self) {
        StdinRouter::global().switch_to(self.target);
    }
}

pub struct SyncREPL<B: Board> {
    editor: Editor<UnboundedBuffer, UnboundedHistory>,
    io: SyncReplIo,
    session: RvdbSession<B>,
    uart_handle: StdinHandle,
    repl_handle: StdinHandle,
    last_line: String,
}

impl<B: Board> SyncREPL<B> {
    pub fn new(session: RvdbSession<B>, uart_handle: StdinHandle) -> Self {
        let (sender, receiver) = mpsc::channel(REPL_INPUT_CAPACITY);
        let router = StdinRouter::global();
        let repl_handle = router.register(sender);
        router.switch_to(repl_handle);

        let mut io = SyncReplIo { input: receiver };
        let editor = EditorBuilder::new_unbounded()
            .with_unbounded_history()
            .build_sync(&mut io)
            .expect("noline editor construction should not perform fallible I/O");

        Self {
            editor,
            io,
            session,
            uart_handle,
            repl_handle,
            last_line: String::new(),
        }
    }

    fn write_all(&mut self, bytes: &[u8]) {
        if let Err(error) = write_crlf(bytes) {
            log::error!("failed to write rvdb output: {error}");
        }
    }

    fn execute_command(&mut self, command: RvdbCommand) -> Result<RvdbCommandResponse, String> {
        if matches!(command, RvdbCommand::Continue { .. }) {
            with_stdin_target(self.uart_handle, self.repl_handle, || {
                self.session.execute_command(command)
            })
        } else {
            self.session.execute_command(command)
        }
    }

    fn execute_line(&mut self, line: &str) -> Result<RvdbCommandResponse, String> {
        let command = match self.session.parse_line(line) {
            Ok(command) => command,
            Err(error) if is_clap_display(&error) => {
                return Ok(RvdbCommandResponse {
                    text: error.to_string(),
                    exit: false,
                });
            }
            Err(error) => return Err(format_clap_error(error)),
        };
        self.execute_command(command)
    }

    pub fn run_script(&mut self, lines: &[String]) -> bool {
        let result = with_stdin_target(self.uart_handle, self.repl_handle, || {
            self.session
                .run_script(lines, |output| write_crlf(output.as_bytes()))
        });
        match result {
            Ok(exit) => exit,
            Err(error) => {
                log::error!("failed to write rvdb output: {error}");
                false
            }
        }
    }

    pub fn run(&mut self) {
        loop {
            let line = match self.editor.readline(super::PROMPT, &mut self.io) {
                Ok(line) => line.trim().to_owned(),
                Err(NolineError::Aborted) => break,
                Err(error) => {
                    self.write_all(format!("Error: {error:?}\r\n").as_bytes());
                    break;
                }
            };

            let Some(line) = resolve_line(&line, &mut self.last_line) else {
                continue;
            };

            match self.execute_line(&line) {
                Ok(response) => {
                    self.write_all(response.text.as_bytes());
                    if response.exit {
                        break;
                    }
                }
                Err(error) => self.write_all(format!("Error: {error}\r\n").as_bytes()),
            }
        }
    }

    pub fn session(&self) -> &RvdbSession<B> {
        &self.session
    }

    pub fn session_mut(&mut self) -> &mut RvdbSession<B> {
        &mut self.session
    }

    pub fn into_board(self) -> B {
        self.session.into_board()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_is_restored_when_continue_returns_error() {
        let router = StdinRouter::global();
        let (uart_tx, _uart_rx) = mpsc::channel(1);
        let (repl_tx, _repl_rx) = mpsc::channel(1);
        let uart = router.register(uart_tx);
        let repl = router.register(repl_tx);
        router.switch_to(repl);

        let result: Result<(), &str> = with_stdin_target(uart, repl, || Err("failed"));

        assert_eq!(result, Err("failed"));
        assert_eq!(router.current_target(), repl);
    }

    #[test]
    fn empty_line_repeats_the_last_command() {
        let mut last = String::new();
        assert_eq!(resolve_line("  ", &mut last), None);
        assert_eq!(resolve_line("step", &mut last), Some("step".to_string()));
        assert_eq!(resolve_line("", &mut last), Some("step".to_string()));
    }
}
