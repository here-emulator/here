//! Just-in-Time (JIT) dynamic binary translator.
//!
//! Compilation is organized as a layered pipeline:
//!
//! ```text
//! Guest Instructions  --(translator)-->  IR Block  --(codegen)-->  Host Instructions
//! ```
//!
//! - **Frontend ([`translator`])**: Translates guest instructions into an [`ir::IRBlock`].
//! - **[IR](`ir`)**: Represents basic blocks in SSA form.
//! - **[Backend](`backend`)**: Generates host machine code.
//! - **[ABI](`abi`)**: Defines CPU state layout and runtime helpers called from JIT code.
//! - **[Runtime](`runtime`)**: Assembles and drives execution, managing block chaining.
//!
//! # Execution
//! - Guest code is translated per basic block, and blocks are linked via block chaining.
//! - Runtime initializes the cycle budget.
//! - Each block begins with a budget check, running continuously until the budget is exhausted.

mod abi;
mod backend;
mod ir;
mod jit_buffer;
mod pass;
mod stats;
mod translator;

use ir::*;

pub mod runtime;
pub use runtime::JitRuntime;
