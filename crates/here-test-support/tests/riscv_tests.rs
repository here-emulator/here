//! Integration tests for the [`riscv-tests`] repo.
//!
//! Every test binary runs on the interpreter and, where the platform provides
//! one, on the JIT executor as well.
//!
//! Build the `riscv-tests` submodule first; see the repository README.
//!
//! [`riscv-tests`]: https://github.com/riscv-software-src/riscv-tests

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use crossterm::style::Stylize;
use here_board::Board;
use here_board::BoardStatus;
use here_board::virt::UartIoMode;
use here_board::virt::VirtBoard;
use here_isa::DebugTarget;
use here_isa::riscv::address::Address;
use here_test_support::board_from_elf;
use here_test_support::jit_board_from_elf;
use here_test_support::jit_supported;
use here_types::WordType;

const RUN_LIMIT: u64 = 10_000_000;

#[derive(Clone, Copy)]
enum Backend {
    Interpreter,
    Jit,
}

enum Outcome {
    Passed,
    Failed,
    TimedOut,
}

impl Backend {
    fn label(self) -> &'static str {
        match self {
            Self::Interpreter => "interpreter",
            Self::Jit => "jit",
        }
    }
}

fn find_tests_exclude(prefix: &str, exclude_names: &[&str]) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let isa_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../riscv-tests/isa");
    if let Ok(entries) = fs::read_dir(&isa_dir) {
        for e in entries.flatten() {
            let path = e.path();
            if !path.is_file() || path.extension().is_some() {
                continue;
            }

            if let Ok(fname) = e.file_name().into_string() {
                if fname.starts_with(prefix)
                    && exclude_names
                        .iter()
                        .all(|&n| (prefix.to_owned() + n) != fname)
                {
                    paths.push(path);
                }
            }
        }
    }
    paths.sort();
    paths.dedup();
    paths
}

fn run_test(elf_path: &Path, backend: Backend) -> bool {
    // Load the ELF file and run it
    let result = std::panic::catch_unwind(|| {
        let bytes = fs::read(elf_path).unwrap();
        let mut board = match backend {
            Backend::Interpreter => board_from_elf(&bytes, UartIoMode::None),
            Backend::Jit => jit_board_from_elf(&bytes, UartIoMode::None),
        }
        .unwrap();
        let tohost: WordType = board.loader().unwrap().get_section_addr(".tohost").unwrap();

        let mut remaining = RUN_LIMIT;
        while board.status() != BoardStatus::Halt {
            if remaining == 0 {
                return Outcome::TimedOut;
            }

            let executed = board
                .step_batch(remaining.min(VirtBoard::STEP_BATCH_CYCLES))
                .cycles;
            remaining -= executed;

            let msg = board.cpu.read_memory::<u64>(Address::Phys(tohost)).unwrap();
            if msg != 0 {
                return if msg == 1 {
                    Outcome::Passed
                } else {
                    Outcome::Failed
                };
            }
        }

        Outcome::Failed
    });

    let width = 48;
    let name = format!("{} [{}]", elf_path.display(), backend.label());

    match result {
        Err(e) => {
            eprintln!("Test {:<width$}{}: {:?}", name, "panicked".red(), e);
            false
        }

        Ok(Outcome::Passed) => {
            eprintln!("Test {:<width$}{}", name, "passed".green());
            true
        }

        Ok(Outcome::Failed) => {
            eprintln!("Test {:<width$}{}", name, "failed".red());
            false
        }

        Ok(Outcome::TimedOut) => {
            eprintln!("Test {:<width$}{}", name, "timedout".red());
            false
        }
    }
}

fn run_test_group(name: &str) {
    run_test_group_exclude(name, &[]);
}

fn run_test_group_exclude(name: &str, exclude_names: &[&str]) {
    let tests = find_tests_exclude(name, exclude_names);
    assert!(
        !tests.is_empty(),
        "No tests named {} found in riscv-tests/isa",
        name
    );

    let mut backends = vec![Backend::Interpreter];
    if jit_supported() {
        backends.push(Backend::Jit);
    }

    let tot = tests.len() * backends.len();

    let mut fail_cnt = 0;
    for elf in &tests {
        for &backend in &backends {
            fail_cnt += !run_test(elf, backend) as u32;
        }
    }

    if fail_cnt > 0 {
        println!("Totally {}/{} runs failed in {}.", fail_cnt, tot, name);
    } else {
        println!("All tests passed in {}.", name);
    }

    assert!(fail_cnt == 0);
}

#[cfg(feature = "riscv64")]
mod rv64_tests {
    use super::*;

    // U mode tests
    #[test]
    fn run_rv64ui() {
        run_test_group_exclude("rv64ui-p-", &["ma_data"]);
        run_test_group_exclude("rv64ui-v-", &["ma_data"]);
    }

    #[test]
    fn run_rv64uf() {
        run_test_group("rv64uf-p-");
        run_test_group("rv64uf-v-");
    }

    #[test]
    fn run_rv64ud() {
        run_test_group("rv64ud-p-");
        run_test_group("rv64ud-v-");
    }

    #[test]
    fn run_rv64ua() {
        run_test_group("rv64ua-p-");
        run_test_group("rv64ua-v-");
    }

    #[test]
    fn run_rv64um() {
        run_test_group("rv64um-p-");
        run_test_group("rv64um-v-");
    }

    // S mode tests
    #[test]
    fn run_rv64si_p() {
        run_test_group_exclude("rv64si-p-", &["sbreak"]);
    }

    // M mode tests
    #[test]
    fn run_rv64mi_p() {
        run_test_group_exclude("rv64mi-p-", &["pmpaddr", "sbreak", "breakpoint"]);
    }

    #[test]
    fn run_rv64uc() {
        run_test_group("rv64uc-p-");
        run_test_group("rv64uc-v-");
    }
}
