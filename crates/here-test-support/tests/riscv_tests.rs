//! Integration tests for the [`riscv-tests`] repo.
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
use here_isa::DebugTarget;
use here_isa::riscv::address::Address;
use here_test_support::board_from_elf;
use here_types::WordType;

fn find_tests_exclude(prefix: &str, exclude_names: &[&str]) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let isa_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../riscv-tests/isa");
    if let Ok(entries) = fs::read_dir(&isa_dir) {
        for e in entries.flatten() {
            if e.path().is_dir() || e.path().extension() != None {
                continue;
            }

            if let Ok(fname) = e.file_name().into_string() {
                if fname.starts_with(prefix)
                    && exclude_names
                        .iter()
                        .all(|&n| (prefix.to_owned() + n) != fname)
                {
                    paths.push(e.path());
                }
            }
        }
    }
    paths.sort();
    paths.dedup();
    paths
}

#[must_use]
fn run_test(elf_path: &Path) -> bool {
    // Load the ELF file and run it
    let result = std::panic::catch_unwind(|| {
        let mut timeout = false;
        let mut run_result = false;
        let mut board = board_from_elf(&fs::read(elf_path).unwrap(), UartIoMode::External)
            .expect("invalid ELF file");
        let tohost: WordType = board.loader().unwrap().get_section_addr(".tohost").unwrap();

        while board.status() != BoardStatus::Halt {
            board.step();

            // Handle tohost
            let instr_cnt = board.cycles();
            if (instr_cnt & (0xFFF)) == 0 {
                let msg = board.cpu.read_memory::<u64>(Address::Phys(tohost)).unwrap();

                if msg != 0 {
                    run_result = msg == 1;
                    break;
                }
            }

            if instr_cnt > 10_000_000 {
                timeout = true;
                break;
            }
        }

        return (run_result, timeout);
    });

    let width = 48;

    match result {
        Err(e) => {
            eprintln!(
                "Test {:<width$}{}: {:?}",
                elf_path.display(),
                "panicked".red(),
                e
            );
            false
        }

        Ok((false, timeout)) => {
            if timeout {
                eprintln!("Test {:<width$}{}", elf_path.display(), "timedout".red());
            } else {
                eprintln!("Test {:<width$}{}", elf_path.display(), "failed".red());
            }
            false
        }

        Ok((true, _)) => {
            eprintln!("Test {:<width$}{}", elf_path.display(), "passed".green());
            true
        }
    }
}

fn run_test_group(name: &str) {
    run_test_group_exclude(name, &[])
}

fn run_test_group_exclude(name: &str, exclude_names: &[&str]) {
    let tests = find_tests_exclude(name, exclude_names);
    assert!(
        !tests.is_empty(),
        "No tests named {} found in riscv-tests/isa",
        name
    );

    let tot = tests.len();

    let mut fail_cnt = 0;
    for elf in tests {
        fail_cnt += !run_test(&elf) as u32;
    }

    if fail_cnt > 0 {
        println!("Totally {}/{} tests failed in {}.", fail_cnt, tot, name);
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
