use super::*;

use gdbstub::common::Signal;
use gdbstub::conn::Connection;
use gdbstub::conn::ConnectionExt;
use gdbstub::stub::DisconnectReason;
use gdbstub::stub::GdbStub;
use gdbstub::stub::SingleThreadStopReason;
use gdbstub::stub::run_blocking;
use gdbstub::target::Target;
use std::net::TcpListener;
use std::net::TcpStream;
#[cfg(unix)]
use std::path::{Path, PathBuf};

use crate::isa::riscv::csr_reg::NamedCsrReg;
use crate::isa::riscv::csr_reg::csr_macro::CSR_NAME;
use crate::isa::riscv::csr_reg::csr_macro::Misa;
#[cfg(unix)]
use std::os::unix::net::UnixListener;
#[cfg(unix)]
use std::os::unix::net::UnixStream;

struct EmuGdbEventLoop<B: Board> {
    _marker: std::marker::PhantomData<B>,
}

impl<B: Board> EmuGdbEventLoop<B> {
    fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: Board> run_blocking::BlockingEventLoop for EmuGdbEventLoop<B> {
    type Target = GdbDebugger<B>;
    type Connection = Box<dyn ConnectionExt<Error = std::io::Error>>;
    type StopReason = SingleThreadStopReason<u64>;

    fn wait_for_stop_reason(
        target: &mut Self::Target,
        conn: &mut Self::Connection,
    ) -> Result<
        run_blocking::Event<SingleThreadStopReason<u64>>,
        run_blocking::WaitForStopReasonError<
            <Self::Target as Target>::Error,
            <Self::Connection as Connection>::Error,
        >,
    > {
        let has_input = || conn.peek().map(|b| b.is_some()).unwrap_or(true);

        let dbg_event = target.run_by_mode_with_batch_check(has_input);

        let event = match dbg_event {
            RunEvent::IncomingData => {
                let byte = conn
                    .read()
                    .map_err(run_blocking::WaitForStopReasonError::Connection)?;

                run_blocking::Event::IncomingData(byte)
            }

            RunEvent::StopReason(debug_event) => {
                let stop_reason = match debug_event {
                    DebugEvent::StepCompleted => SingleThreadStopReason::DoneStep,
                    DebugEvent::BoardHalted => SingleThreadStopReason::Terminated(Signal::SIGSTOP),
                    DebugEvent::BreakpointHit => SingleThreadStopReason::SwBreak(()),
                };

                run_blocking::Event::TargetStopped(stop_reason)
            }
        };

        Ok(event)
    }

    fn on_interrupt(
        _target: &mut Self::Target,
    ) -> Result<Option<SingleThreadStopReason<u64>>, <Self::Target as Target>::Error> {
        Ok(Some(SingleThreadStopReason::Signal(Signal::SIGINT)))
    }
}

fn gdb_stub_event_loop<B: Board>(
    debugger: GdbStub<GdbDebugger<B>, Box<dyn ConnectionExt<Error = std::io::Error>>>,
    mut target: GdbDebugger<B>,
) {
    match debugger.run_blocking::<EmuGdbEventLoop<B>>(&mut target) {
        Ok(disconnect_reason) => match disconnect_reason {
            DisconnectReason::Disconnect => {
                println!("Client disconnected")
            }
            DisconnectReason::TargetExited(code) => {
                println!("Target exited with code {}", code)
            }
            DisconnectReason::TargetTerminated(sig) => {
                println!("Target terminated with signal {}", sig)
            }
            DisconnectReason::Kill => println!("GDB sent a kill command"),
        },
        Err(e) => {
            if e.is_target_error() {
                println!(
                    "target encountered a fatal error: {}",
                    e.into_target_error().unwrap()
                )
            } else if e.is_connection_error() {
                let (e, kind) = e.into_connection_error().unwrap();
                println!("connection error: {:?} - {}", kind, e,)
            } else {
                println!("gdbstub encountered a fatal error: {}", e)
            }
        }
    }
}

type DynResult<T> = Result<T, Box<dyn std::error::Error>>;

fn wait_for_tcp(port: u16) -> DynResult<TcpStream> {
    let sockaddr = format!("127.0.0.1:{}", port);
    eprintln!("Waiting for a GDB connection on {:?}...\r", sockaddr);

    let sock = TcpListener::bind(sockaddr)?;
    let (stream, addr) = sock.accept()?;
    eprintln!("Debugger connected from {}\r", addr);

    Ok(stream)
}

#[cfg(unix)]
fn wait_for_uds(path: &Path) -> DynResult<UnixStream> {
    match std::fs::remove_file(path) {
        Ok(_) => {}
        Err(e) => match e.kind() {
            std::io::ErrorKind::NotFound => {}
            _ => return Err(e.into()),
        },
    }

    eprintln!("Waiting for a GDB connection on {}...", path.display());

    let sock = UnixListener::bind(path)?;
    let (stream, addr) = sock.accept()?;
    eprintln!("Debugger connected from {:?}", addr);

    Ok(stream)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Config {
    Tcp(u16),

    #[cfg(unix)]
    Uds(PathBuf),
}

pub fn event_loop<B: Board>(board: B, cfg: Config) -> DynResult<()> {
    let mut gdb_debugger = GdbDebugger::new(board);

    let misa = gdb_debugger.dbg.read_csr(Misa::get_index()).unwrap();
    let with_f = ((misa >> ('F' as u32 - 'A' as u32)) & 1) != 0;
    let with_d = ((misa >> ('D' as u32 - 'A' as u32)) & 1) != 0;

    let builder =
        desc::DescBuilder::with_csrs(CSR_NAME.entries().map(|(addr, name)| (*addr, *name)));

    let builder = if with_d {
        builder.with_d()
    } else if with_f {
        builder.with_f()
    } else {
        builder
    };

    desc::init_target_desc_xml(builder);

    let conn: Box<dyn ConnectionExt<Error = std::io::Error>> = match cfg {
        Config::Tcp(port) => Box::new(wait_for_tcp(port)?),
        #[cfg(unix)]
        Config::Uds(path) => Box::new(wait_for_uds(&path)?),
    };
    let stub = GdbStub::new(conn);
    gdb_stub_event_loop(stub, gdb_debugger);

    Ok(())
}
