use crate::config::arch_config::WordType;

use super::*;

const CPU_REG: X86Reg = X86Reg::RBP;
const STACK_REG: X86Reg = X86Reg::RSP;

impl TargetInst for X64Inst {
    type Reg = X86Reg;

    #[inline]
    fn emit_to(&self, buf: &mut CodeBuf) {
        self.emit(buf);
    }

    #[inline]
    fn put_asm_text(&self, w: &mut impl std::fmt::Write) -> std::fmt::Result {
        self.put_asm_text(w)
    }

    #[inline]
    fn mov(dst: Self::Reg, src: Self::Reg) -> Self {
        X64Inst::Mov {
            size: OperandSize::Qword,
            dst: dst.into(),
            src: src.into(),
        }
    }

    #[inline]
    fn spill(slot: u32, src: Self::Reg) -> Self {
        let offset = (slot as usize * size_of::<WordType>()) as i32;
        X64Inst::Mov {
            size: OperandSize::Qword,
            dst: RegMem::from_disp(STACK_REG, offset),
            src: src.into(),
        }
    }

    #[inline]
    fn reload(dst: Self::Reg, slot: u32) -> Self {
        let offset = (slot as usize * size_of::<WordType>()) as i32;
        X64Inst::Mov {
            size: OperandSize::Qword,
            dst: dst.into(),
            src: RegMem::from_disp(STACK_REG, offset),
        }
    }
}

pub struct X86CodeGen;

impl<'a> TargetBackend<'a, X86> for X86CodeGen {
    #[inline]
    fn new() -> Self {
        Self {}
    }

    fn codegen(&mut self, _ctx: &mut CodeGenContext<X86>, _insts: &[IRInst]) -> usize {
        todo!()
    }

    fn config(&mut self) -> CodeGenConfig {
        todo!()
    }

    fn emit_stack_frame(&mut self, ctx: &mut CodeGenContext<X86>) -> usize {
        let offset = ctx.buf().len();
        // Fixed 7-byte encoding for `sub rsp, imm32` (REX.W 0x48, Op 0x81, ModRM 0xec, imm32 0x00000000)
        ctx.buf()
            .extend_from_slice(&[0x48, 0x81, 0xec, 0x00, 0x00, 0x00, 0x00]);
        if let Some(asm) = ctx.asm_buf() {
            asm.push_str("sub rsp, <stack_frame>\n");
        }
        offset
    }

    fn patch_stack_frame(&mut self, buf: &mut [u8], offset: usize, slots: u32) {
        let bytes = (slots as usize * size_of::<WordType>()) as u32;
        buf[offset + 3..offset + 7].copy_from_slice(&bytes.to_le_bytes());
    }
}
