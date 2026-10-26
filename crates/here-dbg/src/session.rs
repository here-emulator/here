use super::CommandOutput;
use super::PrintObject;
use super::RvdbCommand;
use super::format_clap_error;
use super::is_clap_display;
use super::printer::Printer;
use crate::debugger::Debugger;
use here_board::Board;
use here_board::load::ELFLoader;
use here_board::load::SymTab;

const PROMPT: &str = "(rvdb) ";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RvdbCommandResponse {
    pub text: String,
    pub exit: bool,
}

pub struct RvdbSession<B: Board> {
    pub(super) dbg: Debugger<B>,
    pub(super) watch_list: Vec<PrintObject>,
    printer: Printer,
}

impl<B: Board> RvdbSession<B> {
    pub fn new(board: B) -> Self {
        Self::with_printer(board, Printer::plain())
    }

    pub fn with_printer(board: B, printer: Printer) -> Self {
        Self {
            dbg: Debugger::new(board),
            watch_list: Vec::new(),
            printer,
        }
    }

    pub fn execute_line(&mut self, line: &str) -> Result<RvdbCommandResponse, String> {
        let line = line.trim();
        if line.is_empty() {
            return Ok(RvdbCommandResponse {
                text: String::new(),
                exit: false,
            });
        }

        let cmd = match self.parse_line(line) {
            Ok(cmd) => cmd,
            Err(error) if is_clap_display(&error) => {
                return Ok(RvdbCommandResponse {
                    text: error.to_string(),
                    exit: false,
                });
            }
            Err(error) => return Err(format_clap_error(error)),
        };
        self.execute_command(cmd)
    }

    /// Run commands in order, forwarding formatted output to `write_output`.
    ///
    /// Returns true when the script executes an exit command.
    pub fn run_script<E>(
        &mut self,
        lines: &[String],
        mut write_output: impl FnMut(&str) -> Result<(), E>,
    ) -> Result<bool, E> {
        for line in lines {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            write_output(&format!("{PROMPT}{line}\n"))?;
            match self.execute_line(line) {
                Ok(response) => {
                    write_output(&response.text)?;
                    if response.exit {
                        return Ok(true);
                    }
                }
                Err(error) => write_output(&format!("Error: {error}\n"))?,
            }
        }
        Ok(false)
    }

    pub fn execute_command(&mut self, cmd: RvdbCommand) -> Result<RvdbCommandResponse, String> {
        let output = self.handle_command(cmd)?;

        Ok(RvdbCommandResponse {
            text: self.printer.render(&output),
            exit: output == CommandOutput::Exit,
        })
    }

    pub fn render_output(&self, output: &CommandOutput) -> String {
        self.printer.render(output)
    }

    pub fn load_symbol_file_bytes(&mut self, bytes: Vec<u8>) -> Result<(), String> {
        let loader = ELFLoader::try_new(bytes).ok_or("Failed to parse ELF file")?;
        if let Some(symtab) = loader.get_symbol_table() {
            self.set_symbol_table(symtab);
            Ok(())
        } else {
            Err("No symbol table found in ELF file".to_string())
        }
    }

    pub fn set_symbol_table(&mut self, symtab: SymTab) {
        self.dbg.set_symbol_table(symtab);
    }

    pub fn board(&self) -> &B {
        self.dbg.board()
    }

    pub fn board_mut(&mut self) -> &mut B {
        self.dbg.board_mut()
    }

    pub fn into_board(self) -> B {
        self.dbg.into_board()
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use super::*;
    use here_board::virt::UartIoMode;
    use here_test_support::empty_board;

    #[test]
    fn run_script_reports_errors_and_stops_on_exit() {
        let board = empty_board(UartIoMode::None);
        let mut session = RvdbSession::new(board);
        let lines = vec![
            "unknown-command".to_owned(),
            String::new(),
            "quit".to_owned(),
            "p pc".to_owned(),
        ];
        let mut output = String::new();

        let exit = session
            .run_script(&lines, |text| {
                output.push_str(text);
                Ok::<(), Infallible>(())
            })
            .unwrap();

        assert!(exit);
        assert!(output.contains("Error:"));
        assert!(output.contains("(rvdb) quit\n"));
        assert!(!output.contains("(rvdb) p pc\n"));
    }
}
