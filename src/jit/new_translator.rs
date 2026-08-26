use crate::{
    config::arch_config::WordType,
    jit::{jit_function::JitContext, old_backend::CodeBuf},
};

use super::*;

trait Translator {
    fn new(context: *mut JitContext) -> Self;
    fn translate(&mut self, start_pc: WordType) -> CodeBuf;
}
