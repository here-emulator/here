use std::cell::RefCell;
use std::collections::VecDeque;
use std::future::poll_fn;
use std::rc::Rc;
use std::task::{Context, Poll};

use embedded_io_async::{ErrorKind, ErrorType, Write};
use noline::{
    async_editor::Editor, builder::EditorBuilder, error::NolineError, history::UnboundedHistory,
    line_buffer::UnboundedBuffer,
};

use crate::board::Board;
use crate::isa::riscv::debugger::DebugEvent;
use crate::rvdb::{Printer, RvdbCommand, RvdbSession, format_clap_error, is_clap_display};

const CONTINUE_CHUNK: u64 = 100_000;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen::prelude::wasm_bindgen)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct REPLResponse {
    pub exit: bool,
    /// Whether `continue` command is canceled.
    pub cancel: bool,
}

struct RvdbChannelState {
    input: VecDeque<u8>,
    output: Vec<u8>,
    cancel: bool,
    read_waker: Option<std::task::Waker>,
}

impl RvdbChannelState {
    pub fn new() -> Self {
        RvdbChannelState {
            input: VecDeque::new().into(),
            output: Vec::new().into(),
            cancel: false,
            read_waker: None.into(),
        }
    }
}

pub struct RvdbChannels;

impl RvdbChannels {
    pub fn new() -> (RvdbChannelTx, RvdbChannelRx) {
        let state = Rc::new(RefCell::new(RvdbChannelState::new()));
        (
            RvdbChannelTx {
                state: state.clone(),
            },
            RvdbChannelRx {
                state: state.clone(),
            },
        )
    }
}

#[derive(Clone)]
pub struct RvdbChannelTx {
    state: Rc<RefCell<RvdbChannelState>>,
}

impl RvdbChannelTx {
    pub fn push_input(&mut self, bytes: &[u8]) {
        let mut state = self.state.borrow_mut();
        state.input.extend(bytes);
        if let Some(waker) = state.read_waker.take() {
            waker.wake();
        }
    }

    pub fn take_output(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.state.borrow_mut().output)
    }

    pub fn cancel_continue(&mut self) {
        self.state.borrow_mut().cancel = true;
    }
}

pub struct RvdbChannelRx {
    state: Rc<RefCell<RvdbChannelState>>,
}

impl RvdbChannelRx {
    fn is_canceled(&self) -> bool {
        self.state.borrow().cancel
    }

    fn set_cancel(&self, value: bool) {
        self.state.borrow_mut().cancel = value;
    }
}

impl ErrorType for RvdbChannelRx {
    type Error = ErrorKind;
}

impl embedded_io_async::Read for RvdbChannelRx {
    async fn read(&mut self, buf: &mut [u8]) -> Result<usize, ErrorKind> {
        if buf.is_empty() {
            return Ok(0);
        }

        poll_fn(|cx: &mut Context<'_>| {
            let mut state = self.state.borrow_mut();

            if !state.input.is_empty() {
                let mut count = 0;
                while count < buf.len() {
                    if let Some(byte) = state.input.pop_front() {
                        buf[count] = byte;
                        count += 1;
                    } else {
                        break;
                    }
                }
                Poll::Ready(Ok(count))
            } else {
                state.read_waker = Some(cx.waker().clone());
                Poll::Pending
            }
        })
        .await
    }
}

impl embedded_io_async::Write for RvdbChannelRx {
    async fn write(&mut self, buf: &[u8]) -> Result<usize, ErrorKind> {
        let mut state = self.state.borrow_mut();
        state.output.extend_from_slice(buf);
        Ok(buf.len())
    }

    async fn flush(&mut self) -> Result<(), ErrorKind> {
        Ok(())
    }
}

/// An async no-std REPL, provided for the WASM environment.
///
/// I/O is async, and long-time tasks (like `continue` command) will yield periodically.
pub struct AsyncREPL<B: Board> {
    editor: Editor<UnboundedBuffer, UnboundedHistory>,
    channel: RvdbChannelRx,
    session: RvdbSession<B>,
    last_line: Option<String>,
}

impl<B: Board> AsyncREPL<B> {
    pub async fn new(board: B, mut channel: RvdbChannelRx) -> Self {
        let editor = EditorBuilder::new_unbounded()
            .with_unbounded_history()
            .build_async(&mut channel)
            .await
            .expect("noline editor construction should not perform fallible IO");

        Self {
            editor,
            channel,
            session: RvdbSession::with_printer(board, Printer::ansi_color()),
            last_line: None,
        }
    }

    pub async fn readline(&mut self) -> Result<String, NolineError> {
        let line = self
            .editor
            .readline(super::PROMPT, &mut self.channel)
            .await?;
        Ok(line.to_owned())
    }

    pub fn line_is_continue_command(&self, line: &str) -> bool {
        let mut line = line.trim();
        if line.is_empty() {
            if let Some(last) = &self.last_line {
                line = last.as_str();
            }
        }

        matches!(
            self.session.parse_line(line),
            Ok(RvdbCommand::Continue { .. })
        )
    }

    pub async fn execute_line(&mut self, line: &str) -> Result<REPLResponse, String> {
        self.execute_line_with_hook(line, |_| {}).await
    }

    async fn run_continue(&mut self, total_steps: u64) -> Result<REPLResponse, String> {
        self.run_continue_with_hook(total_steps, |_| {}).await
    }

    pub async fn run_continue_with_hook<F>(
        &mut self,
        total_steps: u64,
        mut after_chunk: F,
    ) -> Result<REPLResponse, String>
    where
        F: FnMut(&mut RvdbSession<B>),
    {
        let mut remain: u64 = total_steps;

        let event = loop {
            if self.channel.is_canceled() {
                self.channel.set_cancel(false);
                return Ok(REPLResponse {
                    exit: false,
                    cancel: true,
                });
            }

            after_chunk(&mut self.session);

            let current = remain.min(CONTINUE_CHUNK);

            let (event, steps) = self
                .session
                .continue_for_steps(current)
                .map_err(|e| format!("step failed: {}", e))?;

            remain -= steps;

            after_chunk(&mut self.session);

            if event != DebugEvent::StepCompleted || remain == 0 {
                break event;
            }

            yield_once().await;
        };

        if event != DebugEvent::StepCompleted {
            let output = self.session.collect_stop_output(event.clone(), remain)?;
            let output = self.session.render_output(&output);
            self.channel.write(output.as_bytes()).await.unwrap();
        }

        Ok(REPLResponse {
            exit: false,
            cancel: false,
        })
    }

    pub async fn execute_line_with_hook<F>(
        &mut self,
        line: &str,
        after_continue_chunk: F,
    ) -> Result<REPLResponse, String>
    where
        F: FnMut(&mut RvdbSession<B>),
    {
        let mut line = line.trim();
        if line.is_empty() {
            match &self.last_line {
                None => {
                    return Ok(REPLResponse {
                        exit: false,
                        cancel: false,
                    });
                }
                Some(last) => {
                    line = last.as_str();
                }
            }
        } else {
            self.last_line = Some(line.to_owned());
        }

        let cmd = match self.session.parse_line(line) {
            Ok(cmd) => cmd,
            Err(error) if is_clap_display(&error) => {
                self.channel
                    .write(error.to_string().as_bytes())
                    .await
                    .unwrap();
                return Ok(REPLResponse {
                    exit: false,
                    cancel: false,
                });
            }
            Err(error) => return Err(format_clap_error(error)),
        };
        match cmd {
            RvdbCommand::Continue { steps } => {
                self.run_continue_with_hook(steps, after_continue_chunk)
                    .await
            }
            other => {
                let response = self.session.execute_command(other)?;
                self.channel.write(response.text.as_bytes()).await.unwrap();

                Ok(REPLResponse {
                    exit: response.exit,
                    cancel: false,
                })
            }
        }
    }

    pub fn load_symbol_file_bytes(&mut self, bytes: Vec<u8>) -> Result<(), String> {
        self.session.load_symbol_file_bytes(bytes)
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

#[cfg(target_arch = "wasm32")]
fn yield_once() -> impl Future<Output = ()> {
    use js_sys::Promise;
    use wasm_bindgen_futures::JsFuture;

    let promise = Promise::new(&mut |resolve, _| {
        web_sys::window()
            .expect("window not available in WASM")
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve.into(), 0)
            .expect("setTimeout failed");
    });
    async move {
        let _ = JsFuture::from(promise).await;
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn yield_once() -> impl Future<Output = ()> {
    std::future::ready(())
}
