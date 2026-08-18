use crate::{
    config::arch_config::WordType,
    fpu::soft_float::*,
    isa::{
        DebugTarget,
        riscv::{
            csr_reg::{
                PrivilegeLevel,
                csr_macro::{Minstret, Mstatus},
            },
            executor::RVCPU,
            instruction::{
                RVInstrInfo, exec_atomic_function::*, exec_compress_function::*, exec_function::*,
                exec_vector_function::*, instr_table::RiscvInstr,
            },
            trap::{Exception, trap_controller::TrapController},
            vector::arithmetic::*,
        },
    },
};

pub(in crate::isa::riscv) fn get_exec_func(
    instr: RiscvInstr,
) -> fn(RVInstrInfo, &mut RVCPU) -> Result<(), Exception> {
    match instr {
        //---------------------------------------
        // RV_I
        //---------------------------------------

        // Arith
        RiscvInstr::ADD | RiscvInstr::ADDI => exec_arith::<ExecAdd>,
        RiscvInstr::ADDW | RiscvInstr::ADDIW => exec_arith::<ExecAddw>,
        RiscvInstr::SUB => exec_arith::<ExecSub>,
        RiscvInstr::SUBW => exec_arith::<ExecSubw>,

        RiscvInstr::MUL => exec_arith::<ExecMulLow>,
        RiscvInstr::MULH => exec_arith::<ExecMulHighSigned<WordType>>,
        RiscvInstr::MULHU => exec_arith::<ExecMulHighUnsigned<WordType>>,
        RiscvInstr::MULHSU => exec_arith::<ExecMulHighSignedUnsigned>,
        RiscvInstr::DIV => exec_arith::<ExecDivSigned>,
        RiscvInstr::DIVU => exec_arith::<ExecDivUnsigned>,
        RiscvInstr::REM => exec_arith::<ExecRemSigned>,
        RiscvInstr::REMU => exec_arith::<ExecRemUnsigned>,
        RiscvInstr::MULW => exec_arith::<ExecMulw>,
        RiscvInstr::DIVW => exec_arith::<ExecDivw>,
        RiscvInstr::DIVUW => exec_arith::<ExecDivuw>,
        RiscvInstr::REMW => exec_arith::<ExecRemw>,
        RiscvInstr::REMUW => exec_arith::<ExecRemuw>,

        // Shift
        RiscvInstr::SLL | RiscvInstr::SLLI => exec_arith::<ExecSLL>,
        RiscvInstr::SRL | RiscvInstr::SRLI => exec_arith::<ExecSRL>,
        RiscvInstr::SRA | RiscvInstr::SRAI => exec_arith::<ExecSRA>,
        RiscvInstr::SRAW | RiscvInstr::SRAIW => exec_arith::<ExecSRAW>,

        RiscvInstr::SLLW | RiscvInstr::SLLIW => exec_arith::<ExecSLLW>,
        RiscvInstr::SRLW | RiscvInstr::SRLIW => exec_arith::<ExecSRLW>,

        // Cond set
        RiscvInstr::SLT | RiscvInstr::SLTI => exec_arith::<ExecSignedLess>,
        RiscvInstr::SLTU | RiscvInstr::SLTIU => exec_arith::<ExecUnsignedLess>,

        // Bit
        RiscvInstr::AND | RiscvInstr::ANDI => exec_arith::<ExecAnd>,
        RiscvInstr::OR | RiscvInstr::ORI => exec_arith::<ExecOr>,
        RiscvInstr::XOR | RiscvInstr::XORI => exec_arith::<ExecXor>,

        // Branch
        RiscvInstr::BEQ => exec_branch::<ExecEqual>,
        RiscvInstr::BNE => exec_branch::<ExecNotEqual>,
        RiscvInstr::BLT => exec_branch::<ExecSignedLess>,
        RiscvInstr::BGE => exec_branch::<ExecSignedGreatEqual>,
        RiscvInstr::BLTU => exec_branch::<ExecUnsignedLess>,
        RiscvInstr::BGEU => exec_branch::<ExecUnsignedGreatEqual>,

        // Load
        RiscvInstr::LB => exec_load::<u8, true>,
        RiscvInstr::LBU => exec_load::<u8, false>,
        RiscvInstr::LH => exec_load::<u16, true>,
        RiscvInstr::LHU => exec_load::<u16, false>,
        RiscvInstr::LW => exec_load::<u32, true>,
        RiscvInstr::LWU => exec_load::<u32, false>,
        RiscvInstr::LD => exec_load::<u64, false>,

        // Store
        RiscvInstr::SB => exec_store::<u8>,
        RiscvInstr::SH => exec_store::<u16>,
        RiscvInstr::SW => exec_store::<u32>,
        RiscvInstr::SD => exec_store::<u64>,

        // Jump and link
        RiscvInstr::JAL => |inst_info: RVInstrInfo, cpu: &mut RVCPU| {
            if let RVInstrInfo::J { rd, imm } = inst_info {
                let target = cpu.pc.wrapping_add(imm); // imm has been sign_extended

                super::check_jump_alignment(cpu, target)?;

                let link = cpu.pc.wrapping_add(4);
                cpu.reg_file.write(rd, link);
                cpu.pc = target;
            } else {
                std::unreachable!();
            }
            cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);
            Ok(())
        },

        RiscvInstr::JALR => |inst_info: RVInstrInfo, cpu: &mut RVCPU| {
            if let RVInstrInfo::I { rs1, rd, imm } = inst_info {
                let t = cpu.pc + 4;
                let val = cpu.reg_file.read(rs1, 0).0;
                let target: WordType = val.wrapping_add(imm) & !1; // imm has been sign_extended

                super::check_jump_alignment(cpu, target)?;

                cpu.pc = target;
                cpu.reg_file.write(rd, t);
            } else {
                std::unreachable!();
            }

            cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);
            Ok(())
        },

        RiscvInstr::AUIPC => |inst_info, cpu| {
            if let RVInstrInfo::U { rd, imm } = inst_info {
                let value = cpu.pc.wrapping_add(imm); // imm has been sign_extended
                cpu.reg_file.write(rd, value);
                cpu.pc = cpu.pc.wrapping_add(4);
                cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);
                Ok(())
            } else {
                std::unreachable!();
            }
        },

        RiscvInstr::LUI => |inst_info, cpu| {
            if let RVInstrInfo::U { rd, imm } = inst_info {
                cpu.reg_file.write(rd, imm); // imm has been sign_extended
                cpu.pc = cpu.pc.wrapping_add(4);
                cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);
                Ok(())
            } else {
                std::unreachable!();
            }
        },

        RiscvInstr::EBREAK => |_info, _cpu| Err(Exception::Breakpoint),
        RiscvInstr::ECALL => |_info, _cpu| match _cpu.get_current_privilege() {
            PrivilegeLevel::U => Err(Exception::UserEnvCall),
            PrivilegeLevel::S => Err(Exception::SupervisorEnvCall),
            PrivilegeLevel::M => Err(Exception::MachineEnvCall),
            PrivilegeLevel::V => unreachable!(),
        },

        RiscvInstr::FENCE => |info, cpu| {
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            exec_nop(info, cpu)
        },

        RiscvInstr::FENCE_I => |_info, cpu| {
            cpu.flush_icache();
            cpu.flush_tlb();
            cpu.pc = cpu.pc.wrapping_add(4);
            cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);
            Ok(())
        },

        RiscvInstr::CSRRW => exec_csrw::<false>,
        RiscvInstr::CSRRC => exec_csr_bit::<false, false>,
        RiscvInstr::CSRRS => exec_csr_bit::<true, false>,
        RiscvInstr::CSRRWI => exec_csrw::<true>,
        RiscvInstr::CSRRCI => exec_csr_bit::<false, true>,
        RiscvInstr::CSRRSI => exec_csr_bit::<true, true>,

        RiscvInstr::MRET => |_info, cpu| {
            if cpu.get_current_privilege() != PrivilegeLevel::M {
                return Err(Exception::IllegalInstruction);
            }
            TrapController::mret(cpu);

            cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);
            Ok(())
        },
        RiscvInstr::WFI => exec_nop,

        //---------------------------------------
        // RV_F
        //---------------------------------------

        // Arith
        RiscvInstr::FADD_S => exec_float_arith_rm::<f32, AddOp>,
        RiscvInstr::FSUB_S => exec_float_arith_rm::<f32, SubOp>,
        RiscvInstr::FMUL_S => exec_float_arith_rm::<f32, MulOp>,
        RiscvInstr::FDIV_S => exec_float_arith_rm::<f32, DivOp>,
        RiscvInstr::FMADD_S => exec_float_arith_r4_rm::<f32, MulAddOp>,
        RiscvInstr::FNMADD_S => exec_float_arith_r4_rm::<f32, NegMulAddOp>,
        RiscvInstr::FMSUB_S => exec_float_arith_r4_rm::<f32, MulSubOp>,
        RiscvInstr::FNMSUB_S => exec_float_arith_r4_rm::<f32, NegMulSubOp>,
        RiscvInstr::FSQRT_S => exec_sqrt::<f32>,
        RiscvInstr::FMIN_S => exec_float_min::<f32>,
        RiscvInstr::FMAX_S => exec_float_max::<f32>,

        RiscvInstr::FADD_D => exec_float_arith_rm::<f64, AddOp>,
        RiscvInstr::FSUB_D => exec_float_arith_rm::<f64, SubOp>,
        RiscvInstr::FMUL_D => exec_float_arith_rm::<f64, MulOp>,
        RiscvInstr::FDIV_D => exec_float_arith_rm::<f64, DivOp>,
        RiscvInstr::FMADD_D => exec_float_arith_r4_rm::<f64, MulAddOp>,
        RiscvInstr::FNMADD_D => exec_float_arith_r4_rm::<f64, NegMulAddOp>,
        RiscvInstr::FMSUB_D => exec_float_arith_r4_rm::<f64, MulSubOp>,
        RiscvInstr::FNMSUB_D => exec_float_arith_r4_rm::<f64, NegMulSubOp>,
        RiscvInstr::FSQRT_D => exec_sqrt::<f64>,
        RiscvInstr::FMIN_D => exec_float_min::<f64>,
        RiscvInstr::FMAX_D => exec_float_max::<f64>,

        // Sign injection
        RiscvInstr::FSGNJ_S => exec_float_arith::<f32, SignInjectOp>,
        RiscvInstr::FSGNJN_S => exec_float_arith::<f32, SignInjectNegOp>,
        RiscvInstr::FSGNJX_S => exec_float_arith::<f32, SignInjectXorOp>,

        RiscvInstr::FSGNJ_D => exec_float_arith::<f64, SignInjectOp>,
        RiscvInstr::FSGNJN_D => exec_float_arith::<f64, SignInjectNegOp>,
        RiscvInstr::FSGNJX_D => exec_float_arith::<f64, SignInjectXorOp>,

        // Convert (RV32F)
        RiscvInstr::FCVT_W_S => exec_cvt_i_from_f::<f32, u32>,
        RiscvInstr::FCVT_WU_S => exec_cvt_u_from_f::<f32, u32>,
        RiscvInstr::FCVT_S_W => exec_cvt_f_from_i::<f32, 32>,
        RiscvInstr::FCVT_S_WU => exec_cvt_f_from_u::<f32, 32>,

        // Convert (RV64F/D)
        RiscvInstr::FCVT_L_S => exec_cvt_i_from_f::<f32, u64>,
        RiscvInstr::FCVT_LU_S => exec_cvt_u_from_f::<f32, u64>,
        RiscvInstr::FCVT_S_L => exec_cvt_f_from_i::<f32, 64>,
        RiscvInstr::FCVT_S_LU => exec_cvt_f_from_u::<f32, 64>,

        RiscvInstr::FCVT_W_D => exec_cvt_i_from_f::<f64, u32>,
        RiscvInstr::FCVT_WU_D => exec_cvt_u_from_f::<f64, u32>,
        RiscvInstr::FCVT_D_W => exec_cvt_f_from_i::<f64, 32>,
        RiscvInstr::FCVT_D_WU => exec_cvt_f_from_u::<f64, 32>,

        RiscvInstr::FCVT_L_D => exec_cvt_i_from_f::<f64, u64>,
        RiscvInstr::FCVT_LU_D => exec_cvt_u_from_f::<f64, u64>,
        RiscvInstr::FCVT_D_L => exec_cvt_f_from_i::<f64, 64>,
        RiscvInstr::FCVT_D_LU => exec_cvt_f_from_u::<f64, 64>,

        RiscvInstr::FCVT_D_S => exec_cvt_float::<f32, f64>,
        RiscvInstr::FCVT_S_D => exec_cvt_float::<f64, f32>,

        // Float Compare
        RiscvInstr::FEQ_S => exec_float_compare::<EqOp, f32>,
        RiscvInstr::FLT_S => exec_float_compare::<LtOp, f32>,
        RiscvInstr::FLE_S => exec_float_compare::<LeOp, f32>,

        RiscvInstr::FEQ_D => exec_float_compare::<EqOp, f64>,
        RiscvInstr::FLT_D => exec_float_compare::<LtOp, f64>,
        RiscvInstr::FLE_D => exec_float_compare::<LeOp, f64>,

        // Store/Load
        RiscvInstr::FLW => exec_float_load::<f32>,
        RiscvInstr::FSW => exec_float_store::<f32>,

        RiscvInstr::FLD => exec_float_load::<f64>,
        RiscvInstr::FSD => exec_float_store::<f64>,

        // Move
        RiscvInstr::FMV_X_W => exec_mv_x_from_f::<f32, true>,
        RiscvInstr::FMV_W_X => exec_mv_f_from_x::<f32>,

        RiscvInstr::FMV_X_D => exec_mv_x_from_f::<f64, false>,
        RiscvInstr::FMV_D_X => exec_mv_f_from_x::<f64>,

        // Classify
        RiscvInstr::FCLASS_S => exec_float_classify::<f32>,
        RiscvInstr::FCLASS_D => exec_float_classify::<f64>,

        //---------------------------------------
        // RV_A
        //---------------------------------------

        // arith
        RiscvInstr::AMOADD_W => exec_atomic_memory_operation::<ExecAmoAdd, u32>,
        RiscvInstr::AMOAND_W => exec_atomic_memory_operation::<ExecAmoAnd, u32>,
        RiscvInstr::AMOOR_W => exec_atomic_memory_operation::<ExecAmoOr, u32>,
        RiscvInstr::AMOXOR_W => exec_atomic_memory_operation::<ExecAmoXor, u32>,

        RiscvInstr::AMOADD_D => exec_atomic_memory_operation::<ExecAmoAdd, u64>,
        RiscvInstr::AMOAND_D => exec_atomic_memory_operation::<ExecAmoAnd, u64>,
        RiscvInstr::AMOOR_D => exec_atomic_memory_operation::<ExecAmoOr, u64>,
        RiscvInstr::AMOXOR_D => exec_atomic_memory_operation::<ExecAmoXor, u64>,

        // swap
        RiscvInstr::AMOSWAP_W => exec_atomic_memory_operation::<ExecAmoSwap, u32>,
        RiscvInstr::AMOSWAP_D => exec_atomic_memory_operation::<ExecAmoSwap, u64>,

        // cmp
        RiscvInstr::AMOMAX_W => exec_atomic_memory_operation::<ExecAmoMax, u32>,
        RiscvInstr::AMOMIN_W => exec_atomic_memory_operation::<ExecAmoMin, u32>,
        RiscvInstr::AMOMAXU_W => exec_atomic_memory_operation::<ExecAmoMaxU, u32>,
        RiscvInstr::AMOMINU_W => exec_atomic_memory_operation::<ExecAmoMinU, u32>,

        RiscvInstr::AMOMAX_D => exec_atomic_memory_operation::<ExecAmoMax, u64>,
        RiscvInstr::AMOMIN_D => exec_atomic_memory_operation::<ExecAmoMin, u64>,
        RiscvInstr::AMOMAXU_D => exec_atomic_memory_operation::<ExecAmoMaxU, u64>,
        RiscvInstr::AMOMINU_D => exec_atomic_memory_operation::<ExecAmoMinU, u64>,

        // load-reserved / store-conditional
        RiscvInstr::LR_W => exec_lr::<u32, true>,
        RiscvInstr::SC_W => exec_sc::<u32>,

        RiscvInstr::LR_D => exec_lr::<u64, false>,
        RiscvInstr::SC_D => exec_sc::<u64>,

        //---------------------------------------
        // RV_Custom
        //---------------------------------------
        #[cfg(feature = "custom-instr")]
        RiscvInstr::MY_INSTR0_R => todo!(),
        #[cfg(feature = "custom-instr")]
        RiscvInstr::MY_INSTR1_DISPLAY => |info, cpu| {
            if let RVInstrInfo::I { rs1: _, rd: _, imm } = info {
                print!("{}", imm as u8 as char);
            }

            cpu.pc = cpu.pc.wrapping_add(4);
            cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);
            Ok(())
        },

        //---------------------------------------
        // RV_S
        //---------------------------------------
        RiscvInstr::SRET => |_info, cpu| {
            if cpu.get_current_privilege() < PrivilegeLevel::S {
                return Err(Exception::IllegalInstruction);
            }
            if cpu.get_current_privilege() == PrivilegeLevel::S
                && cpu.csr.get_by_type_existing::<Mstatus>().get_tsr() == 1
            {
                return Err(Exception::IllegalInstruction);
            }

            TrapController::sret(cpu);
            cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);
            Ok(())
        },

        RiscvInstr::SFENCE_VMA => |_info, cpu| {
            if cpu.get_current_privilege() < PrivilegeLevel::S {
                return Err(Exception::IllegalInstruction);
            }
            if cpu.get_current_privilege() == PrivilegeLevel::S
                && cpu.csr.get_by_type_existing::<Mstatus>().get_tvm() == 1
            {
                return Err(Exception::IllegalInstruction);
            }

            cpu.memory.flush_tlb();
            cpu.flush_icache();

            cpu.write_pc(cpu.pc.wrapping_add(4));
            cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);
            Ok(())
        },

        //---------------------------------------
        // RV_C
        //---------------------------------------
        RiscvInstr::C_ADD | RiscvInstr::C_ADDI | RiscvInstr::C_ADDI16SP => {
            exec_compress_arith::<ExecAdd>
        }
        RiscvInstr::C_ADDW | RiscvInstr::C_ADDIW => exec_compress_arith::<ExecAddw>,
        RiscvInstr::C_ADDI4SPN => exec_addi4spn,
        RiscvInstr::C_SLLI => exec_compress_arith::<ExecSLL>,
        RiscvInstr::C_SRLI => exec_compress_arith::<ExecSRL>,
        RiscvInstr::C_SRAI => exec_compress_arith::<ExecSRA>,

        RiscvInstr::C_MV => exec_compress_mv,

        RiscvInstr::C_AND | RiscvInstr::C_ANDI => exec_compress_arith::<ExecAnd>,
        RiscvInstr::C_OR => exec_compress_arith::<ExecOr>,
        RiscvInstr::C_XOR => exec_compress_arith::<ExecXor>,
        RiscvInstr::C_SUB => exec_compress_arith::<ExecSub>,
        RiscvInstr::C_SUBW => exec_compress_arith::<ExecSubw>,

        RiscvInstr::C_NOP => exec_compress_nop,
        RiscvInstr::C_EBREAK => |_info, _cpu| Err(Exception::Breakpoint),

        RiscvInstr::C_LI | RiscvInstr::C_LUI => exec_compress_li,

        // Integer load / store
        RiscvInstr::C_LW => exec_compress_load::<u32, true>,
        RiscvInstr::C_LD => exec_compress_load::<u64, false>,
        RiscvInstr::C_LWSP => exec_compress_load_sp::<u32, true>,
        RiscvInstr::C_LDSP => exec_compress_load_sp::<u64, false>,
        RiscvInstr::C_SW => exec_compress_store::<u32>,
        RiscvInstr::C_SD => exec_compress_store::<u64>,
        RiscvInstr::C_SWSP => exec_compress_store_sp::<u32>,
        RiscvInstr::C_SDSP => exec_compress_store_sp::<u64>,

        // Float load / store
        RiscvInstr::C_FLW => exec_compress_float_load::<f32>,
        RiscvInstr::C_FLD => exec_compress_float_load::<f64>,
        RiscvInstr::C_FLWSP => exec_compress_float_load_sp::<f32>,
        RiscvInstr::C_FLDSP => exec_compress_float_load_sp::<f64>,
        RiscvInstr::C_FSW => exec_compress_float_store::<f32>,
        RiscvInstr::C_FSD => exec_compress_float_store::<f64>,
        RiscvInstr::C_FSWSP => exec_compress_float_store_sp::<f32>,
        RiscvInstr::C_FSDSP => exec_compress_float_store_sp::<f64>,

        // Branch
        RiscvInstr::C_BEQZ => exec_compress_branch::<ExecEqual>,
        RiscvInstr::C_BNEZ => exec_compress_branch::<ExecNotEqual>,

        // Jump
        RiscvInstr::C_J => exec_compress_jump::<false>,
        RiscvInstr::C_JAL => exec_compress_jump::<true>,
        RiscvInstr::C_JR => exec_compress_jump_reg::<false>,
        RiscvInstr::C_JALR => exec_compress_jump_reg::<true>,

        // Reserved / canonical illegal encoding (e.g. 0x0000).
        RiscvInstr::ILLEGAL => |_info, _cpu| Err(Exception::IllegalInstruction),

        //---------------------------------------
        // RV_V
        //---------------------------------------
        //--------- config instruction. ---------
        RiscvInstr::VSETVL => exec_vector_config::<VsetvlFieldExtractor>,
        RiscvInstr::VSETVLI => exec_vector_config::<VsetvliFieldExtractor>,
        RiscvInstr::VSETIVLI => exec_vector_config::<VsetivliFieldExtractor>,

        //-------- OPIVV (func3 = 0b000) --------
        RiscvInstr::VADD_VV => vec_integer_op_vv::<VectorOpAdd>, // Single-width Integer Arithmetic Instructions
        RiscvInstr::VSUB_VV => vec_integer_op_vv::<VectorOpSub>,

        RiscvInstr::VMUL_VV => vec_integer_op_vv::<VectorOpMul>, // Single-Width Integer Multiply Instructions
        RiscvInstr::VMULH_VV => vec_integer_op_vv::<VectorOpMulh>,
        RiscvInstr::VMULHU_VV => vec_integer_op_vv::<VectorOpMulhu>,
        RiscvInstr::VMULHSU_VV => vec_integer_op_vv::<VectorOpMulhsu>,

        RiscvInstr::VDIV_VV => vec_integer_op_vv::<VectorOpDiv>, // Integer Divide Instructions
        RiscvInstr::VDIVU_VV => vec_integer_op_vv::<VectorOpDivu>,
        RiscvInstr::VREM_VV => vec_integer_op_vv::<VectorOpRem>,
        RiscvInstr::VREMU_VV => vec_integer_op_vv::<VectorOpRemu>,

        RiscvInstr::VAND_VV => vec_integer_op_vv::<VectorOpAnd>, // Bitwise Logical Instructions
        RiscvInstr::VOR_VV => vec_integer_op_vv::<VectorOpOr>,
        RiscvInstr::VXOR_VV => vec_integer_op_vv::<VectorOpXor>,

        RiscvInstr::VSRA_VV => vec_integer_op_vv::<VectorOpSra>, //  Single-Width Shift Instructions
        RiscvInstr::VSRL_VV => vec_integer_op_vv::<VectorOpSrl>,
        RiscvInstr::VSLL_VV => vec_integer_op_vv::<VectorOpSll>,

        RiscvInstr::VNSRL_WV => vec_integer_spec_op::<{ vector_spec_instr::NSRL_WV }>, // Narrowing Shift Instructions
        RiscvInstr::VNSRA_WV => vec_integer_spec_op::<{ vector_spec_instr::NSRA_WV }>,

        RiscvInstr::VMSEQ_VV => vec_integer_mask_op_vv::<VectorOpMseq>, // Integer Compare Instructions
        RiscvInstr::VMSNE_VV => vec_integer_mask_op_vv::<VectorOpMsne>,
        RiscvInstr::VMSLTU_VV => vec_integer_mask_op_vv::<VectorOpMsltu>,
        RiscvInstr::VMSLT_VV => vec_integer_mask_op_vv::<VectorOpMslt>,
        RiscvInstr::VMSLEU_VV => vec_integer_mask_op_vv::<VectorOpMsleu>,
        RiscvInstr::VMSLE_VV => vec_integer_mask_op_vv::<VectorOpMsle>,

        RiscvInstr::VADC_VVM => vec_integer_op_vvm::<VectorOpAdc>, // Add-with-Carry / Subtract-with-Borrow
        RiscvInstr::VMADC_VV => vec_integer_mask_op_vv::<VectorOpMadc>,
        RiscvInstr::VMADC_VVM => vec_integer_spec_op::<{ vector_spec_instr::MADC_VVM }>,
        RiscvInstr::VSBC_VVM => vec_integer_op_vvm::<VectorOpSbc>,
        RiscvInstr::VMSBC_VV => vec_integer_mask_op_vv::<VectorOpMsbc>,
        RiscvInstr::VMSBC_VVM => vec_integer_spec_op::<{ vector_spec_instr::MSBC_VVM }>,

        RiscvInstr::VMAX_VV => vec_integer_op_vv::<VectorOpMax>, // Integer Min/Max Instructions
        RiscvInstr::VMAXU_VV => vec_integer_op_vv::<VectorOpMaxu>,
        RiscvInstr::VMIN_VV => vec_integer_op_vv::<VectorOpMin>,
        RiscvInstr::VMINU_VV => vec_integer_op_vv::<VectorOpMinu>,

        //  Single-Width Integer Multiply-Add Instructions
        RiscvInstr::VMACC_VV => vec_integer_op_vvv::<VectorOpMacc>, // vd[i] = (vs1[i] * vs2[i]) + vd[i]
        RiscvInstr::VNMSAC_VV => vec_integer_op_vvv::<VectorOpNmsac>, // vd[i] = -(vs1[i] * vs2[i]) + vd[i]
        RiscvInstr::VMADD_VV => vec_integer_op_vvv::<VectorOpMadd>, // vd[i] = (vs1[i] * vd[i]) + vs2[i]
        RiscvInstr::VNMSUB_VV => vec_integer_op_vvv::<VectorOpNmsub>, // vd[i] = -(vs1[i] * vd[i]) + vs2[i]

        RiscvInstr::VMERGE_VVM => vec_integer_op_vvm::<VectorOpMerge>,
        RiscvInstr::VMV_V_V => vec_integer_spec_op::<{ vector_spec_instr::MOVE_V }>,

        RiscvInstr::VSADDU_VV => vec_fixed_point_op_vv::<VectorOpSaddu>, // Single-Width Saturating Add and Subtract
        RiscvInstr::VSADD_VV => vec_fixed_point_op_vv::<VectorOpSadd>,
        RiscvInstr::VSSUBU_VV => vec_fixed_point_op_vv::<VectorOpSsubu>,
        RiscvInstr::VSSUB_VV => vec_fixed_point_op_vv::<VectorOpSsub>,

        RiscvInstr::VAADDU_VV => vec_fixed_point_op_vv::<VectorOpAaddu>, // Single-Width Averaging Add and Subtract
        RiscvInstr::VAADD_VV => vec_fixed_point_op_vv::<VectorOpAadd>,
        RiscvInstr::VASUBU_VV => vec_fixed_point_op_vv::<VectorOpAsubu>,
        RiscvInstr::VASUB_VV => vec_fixed_point_op_vv::<VectorOpAsub>,

        RiscvInstr::VSMUL_VV => vec_fixed_point_op_vv::<VectorOpSmul>, //  Single-Width Fractional Multiply with Rounding and Saturation

        RiscvInstr::VSSRL_VV => vec_fixed_point_op_vv::<VectorOpSsrl>, // Single-Width Scaling Shift Instructions
        RiscvInstr::VSSRA_VV => vec_fixed_point_op_vv::<VectorOpSsra>,

        RiscvInstr::VNCLIPU_WV => vec_fixed_point_narrowing_op_wv::<VectorOpNclipu>, // Narrowing Fixed-Point Clip Instructions
        RiscvInstr::VNCLIP_WV => vec_fixed_point_narrowing_op_wv::<VectorOpNclip>,

        RiscvInstr::VRGATHER_VV => vec_integer_spec_op::<{ vector_spec_instr::GATHER_VV }>, // Vector Gather Instructions
        RiscvInstr::VRGATHEREI16_VV => vec_integer_spec_op::<{ vector_spec_instr::GATHER_EI16_VV }>,

        //-------- OPIVX (0x100) --------
        RiscvInstr::VADD_VX => vec_integer_op_vx::<VectorOpAdd>, // Single-width Integer Arithmetic Instructions
        RiscvInstr::VSUB_VX => vec_integer_op_vx::<VectorOpSub>,
        RiscvInstr::VRSUB_VX => vec_integer_op_vx::<VectorOpRevSub>,

        RiscvInstr::VMUL_VX => vec_integer_op_vx::<VectorOpMul>, // Single-Width Integer Multiply Instructions
        RiscvInstr::VMULH_VX => vec_integer_op_vx::<VectorOpMulh>,
        RiscvInstr::VMULHU_VX => vec_integer_op_vx::<VectorOpMulhu>,
        RiscvInstr::VMULHSU_VX => vec_integer_op_vx::<VectorOpMulhsu>,

        RiscvInstr::VDIV_VX => vec_integer_op_vx::<VectorOpDiv>, // Integer Divide Instructions
        RiscvInstr::VDIVU_VX => vec_integer_op_vx::<VectorOpDivu>,
        RiscvInstr::VREM_VX => vec_integer_op_vx::<VectorOpRem>,
        RiscvInstr::VREMU_VX => vec_integer_op_vx::<VectorOpRemu>,

        RiscvInstr::VAND_VX => vec_integer_op_vx::<VectorOpAnd>, // Bitwise Logical Instructions
        RiscvInstr::VOR_VX => vec_integer_op_vx::<VectorOpOr>,
        RiscvInstr::VXOR_VX => vec_integer_op_vx::<VectorOpXor>,

        RiscvInstr::VSRA_VX => vec_integer_op_vx::<VectorOpSra>, //  Single-Width Shift Instructions
        RiscvInstr::VSRL_VX => vec_integer_op_vx::<VectorOpSrl>,
        RiscvInstr::VSLL_VX => vec_integer_op_vx::<VectorOpSll>,

        RiscvInstr::VNSRL_WX => vec_integer_spec_op::<{ vector_spec_instr::NSRL_WX }>,
        RiscvInstr::VNSRA_WX => vec_integer_spec_op::<{ vector_spec_instr::NSRA_WX }>,

        RiscvInstr::VMSEQ_VX => vec_integer_mask_op_vx::<VectorOpMseq>, // Integer Compare Instructions
        RiscvInstr::VMSNE_VX => vec_integer_mask_op_vx::<VectorOpMsne>,
        RiscvInstr::VMSLTU_VX => vec_integer_mask_op_vx::<VectorOpMsltu>,
        RiscvInstr::VMSLT_VX => vec_integer_mask_op_vx::<VectorOpMslt>,
        RiscvInstr::VMSLEU_VX => vec_integer_mask_op_vx::<VectorOpMsleu>,
        RiscvInstr::VMSLE_VX => vec_integer_mask_op_vx::<VectorOpMsle>,
        RiscvInstr::VMSGTU_VX => vec_integer_mask_op_vx::<VectorOpMsgtu>,
        RiscvInstr::VMSGT_VX => vec_integer_mask_op_vx::<VectorOpMsgt>,

        RiscvInstr::VADC_VXM => vec_integer_op_vxm::<VectorOpAdc>, // Add-with-Carry / Subtract-with-Borrow
        RiscvInstr::VMADC_VX => vec_integer_mask_op_vx::<VectorOpMadc>,
        RiscvInstr::VMADC_VXM => vec_integer_spec_op::<{ vector_spec_instr::MADC_VXM }>,
        RiscvInstr::VSBC_VXM => vec_integer_op_vxm::<VectorOpSbc>,
        RiscvInstr::VMSBC_VX => vec_integer_mask_op_vx::<VectorOpMsbc>,
        RiscvInstr::VMSBC_VXM => vec_integer_spec_op::<{ vector_spec_instr::MSBC_VXM }>,

        RiscvInstr::VMAX_VX => vec_integer_op_vx::<VectorOpMax>, // Integer Min/Max Instructions
        RiscvInstr::VMAXU_VX => vec_integer_op_vx::<VectorOpMaxu>,
        RiscvInstr::VMIN_VX => vec_integer_op_vx::<VectorOpMin>,
        RiscvInstr::VMINU_VX => vec_integer_op_vx::<VectorOpMinu>,

        //  Single-Width Integer Multiply-Add Instructions
        RiscvInstr::VMACC_VX => vec_integer_op_vxv::<VectorOpMacc>, // vd[i] = (x[rs1] * vs2[i]) + vd[i]
        RiscvInstr::VNMSAC_VX => vec_integer_op_vxv::<VectorOpNmsac>, // vd[i] = -(x[rs1] * vs2[i]) + vd[i]
        RiscvInstr::VMADD_VX => vec_integer_op_vxv::<VectorOpMadd>, // vd[i] = (x[rs1] * vd[i]) + vs2[i]
        RiscvInstr::VNMSUB_VX => vec_integer_op_vxv::<VectorOpNmsub>, // vd[i] = -(x[rs1] * vd[i]) + vs2[i]

        RiscvInstr::VMERGE_VXM => vec_integer_op_vxm::<VectorOpMerge>,
        RiscvInstr::VMV_V_X => vec_integer_spec_op::<{ vector_spec_instr::MOVE_VX }>,

        RiscvInstr::VSADDU_VX => vec_fixed_point_op_vx::<VectorOpSaddu>, // Single-Width Saturating Add and Subtract
        RiscvInstr::VSADD_VX => vec_fixed_point_op_vx::<VectorOpSadd>,
        RiscvInstr::VSSUBU_VX => vec_fixed_point_op_vx::<VectorOpSsubu>,
        RiscvInstr::VSSUB_VX => vec_fixed_point_op_vx::<VectorOpSsub>,

        RiscvInstr::VAADDU_VX => vec_fixed_point_op_vx::<VectorOpAaddu>, // Single-Width Averaging Add and Subtract
        RiscvInstr::VAADD_VX => vec_fixed_point_op_vx::<VectorOpAadd>,
        RiscvInstr::VASUBU_VX => vec_fixed_point_op_vx::<VectorOpAsubu>,
        RiscvInstr::VASUB_VX => vec_fixed_point_op_vx::<VectorOpAsub>,

        RiscvInstr::VSMUL_VX => vec_fixed_point_op_vx::<VectorOpSmul>, //  Single-Width Fractional Multiply with Rounding and Saturation

        RiscvInstr::VSSRL_VX => vec_fixed_point_op_vx::<VectorOpSsrl>, // Single-Width Scaling Shift Instructions
        RiscvInstr::VSSRA_VX => vec_fixed_point_op_vx::<VectorOpSsra>,

        RiscvInstr::VNCLIPU_WX => vec_fixed_point_narrowing_op_wx::<VectorOpNclipu>, // Narrowing Fixed-Point Clip Instructions
        RiscvInstr::VNCLIP_WX => vec_fixed_point_narrowing_op_wx::<VectorOpNclip>,

        RiscvInstr::VSLIDEUP_VX => vec_integer_spec_op::<{ vector_spec_instr::SLIDEUP_VX }>, // Vector Slide Instructions
        RiscvInstr::VSLIDEDOWN_VX => vec_integer_spec_op::<{ vector_spec_instr::SLIDEDOWN_VX }>,

        RiscvInstr::VRGATHER_VX => vec_integer_spec_op::<{ vector_spec_instr::GATHER_VX }>, // Vector Gather Instructions

        //-------- OPIVI (func3 = 0b011) --------
        RiscvInstr::VADD_VI => vec_integer_op_vi_signed::<VectorOpAdd>, // Single-width Integer Arithmetic Instructions
        RiscvInstr::VRSUB_VI => vec_integer_op_vi_signed::<VectorOpRevSub>,

        RiscvInstr::VAND_VI => vec_integer_op_vi_signed::<VectorOpAnd>, // Bitwise Logical Instructions
        RiscvInstr::VOR_VI => vec_integer_op_vi_signed::<VectorOpOr>,
        RiscvInstr::VXOR_VI => vec_integer_op_vi_signed::<VectorOpXor>,

        RiscvInstr::VSRA_VI => vec_integer_op_vi_unsigned::<VectorOpSra>, //  Single-Width Shift Instructions
        RiscvInstr::VSRL_VI => vec_integer_op_vi_unsigned::<VectorOpSrl>,
        RiscvInstr::VSLL_VI => vec_integer_op_vi_unsigned::<VectorOpSll>,

        RiscvInstr::VNSRL_WI => vec_integer_spec_op::<{ vector_spec_instr::NSRL_WI }>,
        RiscvInstr::VNSRA_WI => vec_integer_spec_op::<{ vector_spec_instr::NSRA_WI }>,

        RiscvInstr::VMSEQ_VI => vec_integer_mask_op_vi::<VectorOpMseq>, // Integer Compare Instructions
        RiscvInstr::VMSNE_VI => vec_integer_mask_op_vi::<VectorOpMsne>,
        RiscvInstr::VMSLEU_VI => vec_integer_mask_op_vi::<VectorOpMsleu>,
        RiscvInstr::VMSLE_VI => vec_integer_mask_op_vi::<VectorOpMsle>,
        RiscvInstr::VMSGTU_VI => vec_integer_mask_op_vi::<VectorOpMsgtu>,
        RiscvInstr::VMSGT_VI => vec_integer_mask_op_vi::<VectorOpMsgt>,

        RiscvInstr::VADC_VIM => vec_integer_spec_op::<{ vector_spec_instr::ADC_VIM }>, // Add-with-Carry / Subtract-with-Borrow
        RiscvInstr::VMADC_VI => vec_integer_mask_op_vi::<VectorOpMadc>,
        RiscvInstr::VMADC_VIM => vec_integer_spec_op::<{ vector_spec_instr::MADC_VIM }>,

        RiscvInstr::VMERGE_VIM => vec_integer_spec_op::<{ vector_spec_instr::MERGE_VIM }>,
        RiscvInstr::VMV_V_I => vec_integer_spec_op::<{ vector_spec_instr::MOVE_VI }>,

        RiscvInstr::VSADDU_VI => vec_fixed_point_op_vi::<VectorOpSaddu, true>, // Single-Width Saturating Add and Subtract
        RiscvInstr::VSADD_VI => vec_fixed_point_op_vi::<VectorOpSadd, true>,

        RiscvInstr::VSSRL_VI => vec_fixed_point_op_vi::<VectorOpSsrl, false>, // Single-Width Scaling Shift Instructions
        RiscvInstr::VSSRA_VI => vec_fixed_point_op_vi::<VectorOpSsra, false>,

        RiscvInstr::VNCLIPU_WI => vec_fixed_point_narrowing_op_wi::<VectorOpNclipu>, // Narrowing Fixed-Point Clip Instructions
        RiscvInstr::VNCLIP_WI => vec_fixed_point_narrowing_op_wi::<VectorOpNclip>,

        RiscvInstr::VSLIDEUP_VI => vec_integer_spec_op::<{ vector_spec_instr::SLIDEUP_VI }>, // Vector Slide Instructions
        RiscvInstr::VSLIDEDOWN_VI => vec_integer_spec_op::<{ vector_spec_instr::SLIDEDOWN_VI }>,

        RiscvInstr::VRGATHER_VI => vec_integer_spec_op::<{ vector_spec_instr::GATHER_VI }>, // Vector Gather Instructions

        RiscvInstr::VMV1R_V => vec_whole_register_move_op_v::<1>,
        RiscvInstr::VMV2R_V => vec_whole_register_move_op_v::<2>,
        RiscvInstr::VMV4R_V => vec_whole_register_move_op_v::<4>,
        RiscvInstr::VMV8R_V => vec_whole_register_move_op_v::<8>,

        //-------- OPMVV (func3 = 0b010) --------
        RiscvInstr::VWADD_VV => vec_widening_integer_op_vv::<VectorOpWadd>, // Widening Integer Add/Subtract
        RiscvInstr::VWADD_WV => vec_widening_integer_op_wv::<VectorOpWadd>,
        RiscvInstr::VWADDU_VV => vec_widening_integer_op_vv::<VectorOpWaddu>,
        RiscvInstr::VWADDU_WV => vec_widening_integer_op_wv::<VectorOpWaddu>,
        RiscvInstr::VWSUB_VV => vec_widening_integer_op_vv::<VectorOpWsub>,
        RiscvInstr::VWSUB_WV => vec_widening_integer_op_wv::<VectorOpWsub>,
        RiscvInstr::VWSUBU_VV => vec_widening_integer_op_vv::<VectorOpWsubu>,
        RiscvInstr::VWSUBU_WV => vec_widening_integer_op_wv::<VectorOpWsubu>,

        RiscvInstr::VWMUL_VV => vec_widening_integer_op_vv::<VectorOpWmul>, // Widening Integer Multiply Instructions
        RiscvInstr::VWMULU_VV => vec_widening_integer_op_vv::<VectorOpWmulu>,
        RiscvInstr::VWMULSU_VV => vec_widening_integer_op_vv::<VectorOpWmulsu>,

        RiscvInstr::VZEXT_VF2 => vec_integer_ext_op_v::<VectorOpZextVf2, 2>,
        RiscvInstr::VZEXT_VF4 => vec_integer_ext_op_v::<VectorOpZextVf4, 4>,
        RiscvInstr::VZEXT_VF8 => vec_integer_ext_op_v::<VectorOpZextVf8, 8>,
        RiscvInstr::VSEXT_VF2 => vec_integer_ext_op_v::<VectorOpSextVf2, 2>,
        RiscvInstr::VSEXT_VF4 => vec_integer_ext_op_v::<VectorOpSextVf4, 4>,
        RiscvInstr::VSEXT_VF8 => vec_integer_ext_op_v::<VectorOpSextVf8, 8>,

        RiscvInstr::VWMACCU_VV => vec_widening_integer_op_vvv::<VectorOpWmaccu>, // Widening Integer Multiply-Add Instructions
        RiscvInstr::VWMACC_VV => vec_widening_integer_op_vvv::<VectorOpWmacc>,
        RiscvInstr::VWMACCSU_VV => vec_widening_integer_op_vvv::<VectorOpWmaccsu>,

        RiscvInstr::VCOMPRESS_VM => vec_integer_spec_op::<{ vector_spec_instr::COMPRESS_VM }>, // Vector Compress, Expand, and Slide Instructions

        RiscvInstr::VMAND_MM => vec_bit_op_vv::<VectorOpAnd>, // Mask-Register Logical Instructions
        RiscvInstr::VMNAND_MM => vec_bit_op_vv::<VectorOpNand>,
        RiscvInstr::VMANDN_MM => vec_bit_op_vv::<VectorOpAndn>,
        RiscvInstr::VMXOR_MM => vec_bit_op_vv::<VectorOpXor>,
        RiscvInstr::VMOR_MM => vec_bit_op_vv::<VectorOpOr>,
        RiscvInstr::VMNOR_MM => vec_bit_op_vv::<VectorOpNor>,
        RiscvInstr::VMORN_MM => vec_bit_op_vv::<VectorOpOrn>,
        RiscvInstr::VMXNOR_MM => vec_bit_op_vv::<VectorOpXnor>,

        RiscvInstr::VCPOP_M => vec_integer_spec_op::<{ vector_spec_instr::CPOP_M }>, // count population in mask vcpop.m
        RiscvInstr::VFIRST_M => vec_integer_spec_op::<{ vector_spec_instr::FIRST_M }>, // find first set bit in mask vfirst.m
        RiscvInstr::VMSBF_M => vec_integer_spec_op::<{ vector_spec_instr::MSBF_M }>, // set-before-first mask bit
        RiscvInstr::VMSIF_M => vec_integer_spec_op::<{ vector_spec_instr::MSIF_M }>, // set-including-first mask bit
        RiscvInstr::VMSOF_M => vec_integer_spec_op::<{ vector_spec_instr::MSOF_M }>, // set-only-first mask bit
        RiscvInstr::VIOTA_M => vec_integer_spec_op::<{ vector_spec_instr::IOTA_M }>, // Iota Instruction
        RiscvInstr::VID_V => vec_integer_spec_op::<{ vector_spec_instr::ID_V }>, // Element Index Instruction

        RiscvInstr::VMV_X_S => vec_integer_spec_op::<{ vector_spec_instr::MOVE_XS }>, // Vector Scalar Move Instructions
        RiscvInstr::VMV_S_X => vec_integer_spec_op::<{ vector_spec_instr::MOVE_SX }>,
        // RiscvInstr::VFMV_S_F => unimplemented!(),
        // RiscvInstr::VFMV_F_S => unimplemented!(),

        //-------- OPMVX (func3 = 0b110) --------
        RiscvInstr::VWADD_VX => vec_widening_integer_op_vx::<VectorOpWadd>, // Widening Integer Add/Subtract
        RiscvInstr::VWADD_WX => vec_widening_integer_op_wx::<VectorOpWadd>,
        RiscvInstr::VWADDU_VX => vec_widening_integer_op_vx::<VectorOpWaddu>,
        RiscvInstr::VWADDU_WX => vec_widening_integer_op_wx::<VectorOpWaddu>,
        RiscvInstr::VWSUB_VX => vec_widening_integer_op_vx::<VectorOpWsub>,
        RiscvInstr::VWSUB_WX => vec_widening_integer_op_wx::<VectorOpWsub>,
        RiscvInstr::VWSUBU_VX => vec_widening_integer_op_vx::<VectorOpWsubu>,
        RiscvInstr::VWSUBU_WX => vec_widening_integer_op_wx::<VectorOpWsubu>,

        RiscvInstr::VWMUL_VX => vec_widening_integer_op_vx::<VectorOpWmul>, // Widening Integer Multiply Instructions
        RiscvInstr::VWMULU_VX => vec_widening_integer_op_vx::<VectorOpWmulu>,
        RiscvInstr::VWMULSU_VX => vec_widening_integer_op_vx::<VectorOpWmulsu>,

        RiscvInstr::VWMACCU_VX => vec_widening_integer_op_vxv::<VectorOpWmaccu>, // Widening Integer Multiply-Add Instructions
        RiscvInstr::VWMACC_VX => vec_widening_integer_op_vxv::<VectorOpWmacc>,
        RiscvInstr::VWMACCSU_VX => vec_widening_integer_op_vxv::<VectorOpWmaccsu>,
        RiscvInstr::VWMACCUS_VX => vec_widening_integer_op_vxv::<VectorOpWmaccus>,
        //-------- OPFVV (func3 = 0b001) --------
        //-------- OPFVF (func3 = 0b101) --------

        //------- load store instruction. -------
        RiscvInstr::VLM_V => vector_mask_load::<0>,
        RiscvInstr::VSM_V => vector_mask_store::<0>,

        // Vector unit-stride loads: vl[seg]e{eew}.v
        RiscvInstr::VLE8_V
        | RiscvInstr::VLSEG2E8_V
        | RiscvInstr::VLSEG3E8_V
        | RiscvInstr::VLSEG4E8_V
        | RiscvInstr::VLSEG5E8_V
        | RiscvInstr::VLSEG6E8_V
        | RiscvInstr::VLSEG7E8_V
        | RiscvInstr::VLSEG8E8_V => vector_unit_stride_load::<0>,

        RiscvInstr::VLE16_V
        | RiscvInstr::VLSEG2E16_V
        | RiscvInstr::VLSEG3E16_V
        | RiscvInstr::VLSEG4E16_V
        | RiscvInstr::VLSEG5E16_V
        | RiscvInstr::VLSEG6E16_V
        | RiscvInstr::VLSEG7E16_V
        | RiscvInstr::VLSEG8E16_V => vector_unit_stride_load::<1>,

        RiscvInstr::VLE32_V
        | RiscvInstr::VLSEG2E32_V
        | RiscvInstr::VLSEG3E32_V
        | RiscvInstr::VLSEG4E32_V
        | RiscvInstr::VLSEG5E32_V
        | RiscvInstr::VLSEG6E32_V
        | RiscvInstr::VLSEG7E32_V
        | RiscvInstr::VLSEG8E32_V => vector_unit_stride_load::<2>,

        RiscvInstr::VLE64_V
        | RiscvInstr::VLSEG2E64_V
        | RiscvInstr::VLSEG3E64_V
        | RiscvInstr::VLSEG4E64_V
        | RiscvInstr::VLSEG5E64_V
        | RiscvInstr::VLSEG6E64_V
        | RiscvInstr::VLSEG7E64_V
        | RiscvInstr::VLSEG8E64_V => vector_unit_stride_load::<3>,

        // Vector unit-stride fault-only-first loads: vl[seg]e{eew}ff.v
        RiscvInstr::VLE8FF_V
        | RiscvInstr::VLSEG2E8FF_V
        | RiscvInstr::VLSEG3E8FF_V
        | RiscvInstr::VLSEG4E8FF_V
        | RiscvInstr::VLSEG5E8FF_V
        | RiscvInstr::VLSEG6E8FF_V
        | RiscvInstr::VLSEG7E8FF_V
        | RiscvInstr::VLSEG8E8FF_V => vector_unit_stride_fault_only_first_load::<0>,

        RiscvInstr::VLE16FF_V
        | RiscvInstr::VLSEG2E16FF_V
        | RiscvInstr::VLSEG3E16FF_V
        | RiscvInstr::VLSEG4E16FF_V
        | RiscvInstr::VLSEG5E16FF_V
        | RiscvInstr::VLSEG6E16FF_V
        | RiscvInstr::VLSEG7E16FF_V
        | RiscvInstr::VLSEG8E16FF_V => vector_unit_stride_fault_only_first_load::<1>,

        RiscvInstr::VLE32FF_V
        | RiscvInstr::VLSEG2E32FF_V
        | RiscvInstr::VLSEG3E32FF_V
        | RiscvInstr::VLSEG4E32FF_V
        | RiscvInstr::VLSEG5E32FF_V
        | RiscvInstr::VLSEG6E32FF_V
        | RiscvInstr::VLSEG7E32FF_V
        | RiscvInstr::VLSEG8E32FF_V => vector_unit_stride_fault_only_first_load::<2>,

        RiscvInstr::VLE64FF_V
        | RiscvInstr::VLSEG2E64FF_V
        | RiscvInstr::VLSEG3E64FF_V
        | RiscvInstr::VLSEG4E64FF_V
        | RiscvInstr::VLSEG5E64FF_V
        | RiscvInstr::VLSEG6E64FF_V
        | RiscvInstr::VLSEG7E64FF_V
        | RiscvInstr::VLSEG8E64FF_V => vector_unit_stride_fault_only_first_load::<3>,

        // Vector constant-stride loads: vls[seg]e{eew}.v
        RiscvInstr::VLSE8_V
        | RiscvInstr::VLSSEG2E8_V
        | RiscvInstr::VLSSEG3E8_V
        | RiscvInstr::VLSSEG4E8_V
        | RiscvInstr::VLSSEG5E8_V
        | RiscvInstr::VLSSEG6E8_V
        | RiscvInstr::VLSSEG7E8_V
        | RiscvInstr::VLSSEG8E8_V => vector_constant_stride_load::<0>,

        RiscvInstr::VLSE16_V
        | RiscvInstr::VLSSEG2E16_V
        | RiscvInstr::VLSSEG3E16_V
        | RiscvInstr::VLSSEG4E16_V
        | RiscvInstr::VLSSEG5E16_V
        | RiscvInstr::VLSSEG6E16_V
        | RiscvInstr::VLSSEG7E16_V
        | RiscvInstr::VLSSEG8E16_V => vector_constant_stride_load::<1>,

        RiscvInstr::VLSE32_V
        | RiscvInstr::VLSSEG2E32_V
        | RiscvInstr::VLSSEG3E32_V
        | RiscvInstr::VLSSEG4E32_V
        | RiscvInstr::VLSSEG5E32_V
        | RiscvInstr::VLSSEG6E32_V
        | RiscvInstr::VLSSEG7E32_V
        | RiscvInstr::VLSSEG8E32_V => vector_constant_stride_load::<2>,

        RiscvInstr::VLSE64_V
        | RiscvInstr::VLSSEG2E64_V
        | RiscvInstr::VLSSEG3E64_V
        | RiscvInstr::VLSSEG4E64_V
        | RiscvInstr::VLSSEG5E64_V
        | RiscvInstr::VLSSEG6E64_V
        | RiscvInstr::VLSSEG7E64_V
        | RiscvInstr::VLSSEG8E64_V => vector_constant_stride_load::<3>,

        // Vector indexed-unordered loads: vlux[seg]ei{eew}.v
        RiscvInstr::VLUXEI8_V
        | RiscvInstr::VLUXSEG2EI8_V
        | RiscvInstr::VLUXSEG3EI8_V
        | RiscvInstr::VLUXSEG4EI8_V
        | RiscvInstr::VLUXSEG5EI8_V
        | RiscvInstr::VLUXSEG6EI8_V
        | RiscvInstr::VLUXSEG7EI8_V
        | RiscvInstr::VLUXSEG8EI8_V => vector_indexed_unordered_load::<0>,

        RiscvInstr::VLUXEI16_V
        | RiscvInstr::VLUXSEG2EI16_V
        | RiscvInstr::VLUXSEG3EI16_V
        | RiscvInstr::VLUXSEG4EI16_V
        | RiscvInstr::VLUXSEG5EI16_V
        | RiscvInstr::VLUXSEG6EI16_V
        | RiscvInstr::VLUXSEG7EI16_V
        | RiscvInstr::VLUXSEG8EI16_V => vector_indexed_unordered_load::<1>,

        RiscvInstr::VLUXEI32_V
        | RiscvInstr::VLUXSEG2EI32_V
        | RiscvInstr::VLUXSEG3EI32_V
        | RiscvInstr::VLUXSEG4EI32_V
        | RiscvInstr::VLUXSEG5EI32_V
        | RiscvInstr::VLUXSEG6EI32_V
        | RiscvInstr::VLUXSEG7EI32_V
        | RiscvInstr::VLUXSEG8EI32_V => vector_indexed_unordered_load::<2>,

        RiscvInstr::VLUXEI64_V
        | RiscvInstr::VLUXSEG2EI64_V
        | RiscvInstr::VLUXSEG3EI64_V
        | RiscvInstr::VLUXSEG4EI64_V
        | RiscvInstr::VLUXSEG5EI64_V
        | RiscvInstr::VLUXSEG6EI64_V
        | RiscvInstr::VLUXSEG7EI64_V
        | RiscvInstr::VLUXSEG8EI64_V => vector_indexed_unordered_load::<3>,

        // Vector indexed-ordered loads: vlox[seg]ei{eew}.v
        RiscvInstr::VLOXEI8_V
        | RiscvInstr::VLOXSEG2EI8_V
        | RiscvInstr::VLOXSEG3EI8_V
        | RiscvInstr::VLOXSEG4EI8_V
        | RiscvInstr::VLOXSEG5EI8_V
        | RiscvInstr::VLOXSEG6EI8_V
        | RiscvInstr::VLOXSEG7EI8_V
        | RiscvInstr::VLOXSEG8EI8_V => vector_indexed_ordered_load::<0>,

        RiscvInstr::VLOXEI16_V
        | RiscvInstr::VLOXSEG2EI16_V
        | RiscvInstr::VLOXSEG3EI16_V
        | RiscvInstr::VLOXSEG4EI16_V
        | RiscvInstr::VLOXSEG5EI16_V
        | RiscvInstr::VLOXSEG6EI16_V
        | RiscvInstr::VLOXSEG7EI16_V
        | RiscvInstr::VLOXSEG8EI16_V => vector_indexed_ordered_load::<1>,

        RiscvInstr::VLOXEI32_V
        | RiscvInstr::VLOXSEG2EI32_V
        | RiscvInstr::VLOXSEG3EI32_V
        | RiscvInstr::VLOXSEG4EI32_V
        | RiscvInstr::VLOXSEG5EI32_V
        | RiscvInstr::VLOXSEG6EI32_V
        | RiscvInstr::VLOXSEG7EI32_V
        | RiscvInstr::VLOXSEG8EI32_V => vector_indexed_ordered_load::<2>,

        RiscvInstr::VLOXEI64_V
        | RiscvInstr::VLOXSEG2EI64_V
        | RiscvInstr::VLOXSEG3EI64_V
        | RiscvInstr::VLOXSEG4EI64_V
        | RiscvInstr::VLOXSEG5EI64_V
        | RiscvInstr::VLOXSEG6EI64_V
        | RiscvInstr::VLOXSEG7EI64_V
        | RiscvInstr::VLOXSEG8EI64_V => vector_indexed_ordered_load::<3>,

        // Vector load whole register: vl{nf}re{eew}.v
        RiscvInstr::VL1RE8_V
        | RiscvInstr::VL2RE8_V
        | RiscvInstr::VL4RE8_V
        | RiscvInstr::VL8RE8_V => vector_whole_register_load::<0>,

        RiscvInstr::VL1RE16_V
        | RiscvInstr::VL2RE16_V
        | RiscvInstr::VL4RE16_V
        | RiscvInstr::VL8RE16_V => vector_whole_register_load::<1>,

        RiscvInstr::VL1RE32_V
        | RiscvInstr::VL2RE32_V
        | RiscvInstr::VL4RE32_V
        | RiscvInstr::VL8RE32_V => vector_whole_register_load::<2>,

        RiscvInstr::VL1RE64_V
        | RiscvInstr::VL2RE64_V
        | RiscvInstr::VL4RE64_V
        | RiscvInstr::VL8RE64_V => vector_whole_register_load::<3>,

        RiscvInstr::VS1R_V | RiscvInstr::VS2R_V | RiscvInstr::VS4R_V | RiscvInstr::VS8R_V => {
            vector_whole_register_store::<0>
        }

        // Vector unit-stride stores: vs[seg]e{eew}.v
        RiscvInstr::VSE8_V
        | RiscvInstr::VSSEG2E8_V
        | RiscvInstr::VSSEG3E8_V
        | RiscvInstr::VSSEG4E8_V
        | RiscvInstr::VSSEG5E8_V
        | RiscvInstr::VSSEG6E8_V
        | RiscvInstr::VSSEG7E8_V
        | RiscvInstr::VSSEG8E8_V => vector_unit_stride_store::<0>,

        RiscvInstr::VSE16_V
        | RiscvInstr::VSSEG2E16_V
        | RiscvInstr::VSSEG3E16_V
        | RiscvInstr::VSSEG4E16_V
        | RiscvInstr::VSSEG5E16_V
        | RiscvInstr::VSSEG6E16_V
        | RiscvInstr::VSSEG7E16_V
        | RiscvInstr::VSSEG8E16_V => vector_unit_stride_store::<1>,

        RiscvInstr::VSE32_V
        | RiscvInstr::VSSEG2E32_V
        | RiscvInstr::VSSEG3E32_V
        | RiscvInstr::VSSEG4E32_V
        | RiscvInstr::VSSEG5E32_V
        | RiscvInstr::VSSEG6E32_V
        | RiscvInstr::VSSEG7E32_V
        | RiscvInstr::VSSEG8E32_V => vector_unit_stride_store::<2>,

        RiscvInstr::VSE64_V
        | RiscvInstr::VSSEG2E64_V
        | RiscvInstr::VSSEG3E64_V
        | RiscvInstr::VSSEG4E64_V
        | RiscvInstr::VSSEG5E64_V
        | RiscvInstr::VSSEG6E64_V
        | RiscvInstr::VSSEG7E64_V
        | RiscvInstr::VSSEG8E64_V => vector_unit_stride_store::<3>,

        // Vector constant-stride stores: vss[seg]e{eew}.v
        RiscvInstr::VSSE8_V
        | RiscvInstr::VSSSEG2E8_V
        | RiscvInstr::VSSSEG3E8_V
        | RiscvInstr::VSSSEG4E8_V
        | RiscvInstr::VSSSEG5E8_V
        | RiscvInstr::VSSSEG6E8_V
        | RiscvInstr::VSSSEG7E8_V
        | RiscvInstr::VSSSEG8E8_V => vector_constant_stride_store::<0>,

        RiscvInstr::VSSE16_V
        | RiscvInstr::VSSSEG2E16_V
        | RiscvInstr::VSSSEG3E16_V
        | RiscvInstr::VSSSEG4E16_V
        | RiscvInstr::VSSSEG5E16_V
        | RiscvInstr::VSSSEG6E16_V
        | RiscvInstr::VSSSEG7E16_V
        | RiscvInstr::VSSSEG8E16_V => vector_constant_stride_store::<1>,

        RiscvInstr::VSSE32_V
        | RiscvInstr::VSSSEG2E32_V
        | RiscvInstr::VSSSEG3E32_V
        | RiscvInstr::VSSSEG4E32_V
        | RiscvInstr::VSSSEG5E32_V
        | RiscvInstr::VSSSEG6E32_V
        | RiscvInstr::VSSSEG7E32_V
        | RiscvInstr::VSSSEG8E32_V => vector_constant_stride_store::<2>,

        RiscvInstr::VSSE64_V
        | RiscvInstr::VSSSEG2E64_V
        | RiscvInstr::VSSSEG3E64_V
        | RiscvInstr::VSSSEG4E64_V
        | RiscvInstr::VSSSEG5E64_V
        | RiscvInstr::VSSSEG6E64_V
        | RiscvInstr::VSSSEG7E64_V
        | RiscvInstr::VSSSEG8E64_V => vector_constant_stride_store::<3>,

        // Vector indexed-unordered stores: vsux[seg]ei{eew}.v
        RiscvInstr::VSUXEI8_V
        | RiscvInstr::VSUXSEG2EI8_V
        | RiscvInstr::VSUXSEG3EI8_V
        | RiscvInstr::VSUXSEG4EI8_V
        | RiscvInstr::VSUXSEG5EI8_V
        | RiscvInstr::VSUXSEG6EI8_V
        | RiscvInstr::VSUXSEG7EI8_V
        | RiscvInstr::VSUXSEG8EI8_V => vector_indexed_unordered_store::<0>,

        RiscvInstr::VSUXEI16_V
        | RiscvInstr::VSUXSEG2EI16_V
        | RiscvInstr::VSUXSEG3EI16_V
        | RiscvInstr::VSUXSEG4EI16_V
        | RiscvInstr::VSUXSEG5EI16_V
        | RiscvInstr::VSUXSEG6EI16_V
        | RiscvInstr::VSUXSEG7EI16_V
        | RiscvInstr::VSUXSEG8EI16_V => vector_indexed_unordered_store::<1>,

        RiscvInstr::VSUXEI32_V
        | RiscvInstr::VSUXSEG2EI32_V
        | RiscvInstr::VSUXSEG3EI32_V
        | RiscvInstr::VSUXSEG4EI32_V
        | RiscvInstr::VSUXSEG5EI32_V
        | RiscvInstr::VSUXSEG6EI32_V
        | RiscvInstr::VSUXSEG7EI32_V
        | RiscvInstr::VSUXSEG8EI32_V => vector_indexed_unordered_store::<2>,

        RiscvInstr::VSUXEI64_V
        | RiscvInstr::VSUXSEG2EI64_V
        | RiscvInstr::VSUXSEG3EI64_V
        | RiscvInstr::VSUXSEG4EI64_V
        | RiscvInstr::VSUXSEG5EI64_V
        | RiscvInstr::VSUXSEG6EI64_V
        | RiscvInstr::VSUXSEG7EI64_V
        | RiscvInstr::VSUXSEG8EI64_V => vector_indexed_unordered_store::<3>,

        // Vector indexed-ordered stores: vsox[seg]ei{eew}.v
        RiscvInstr::VSOXEI8_V
        | RiscvInstr::VSOXSEG2EI8_V
        | RiscvInstr::VSOXSEG3EI8_V
        | RiscvInstr::VSOXSEG4EI8_V
        | RiscvInstr::VSOXSEG5EI8_V
        | RiscvInstr::VSOXSEG6EI8_V
        | RiscvInstr::VSOXSEG7EI8_V
        | RiscvInstr::VSOXSEG8EI8_V => vector_indexed_ordered_store::<0>,

        RiscvInstr::VSOXEI16_V
        | RiscvInstr::VSOXSEG2EI16_V
        | RiscvInstr::VSOXSEG3EI16_V
        | RiscvInstr::VSOXSEG4EI16_V
        | RiscvInstr::VSOXSEG5EI16_V
        | RiscvInstr::VSOXSEG6EI16_V
        | RiscvInstr::VSOXSEG7EI16_V
        | RiscvInstr::VSOXSEG8EI16_V => vector_indexed_ordered_store::<1>,

        RiscvInstr::VSOXEI32_V
        | RiscvInstr::VSOXSEG2EI32_V
        | RiscvInstr::VSOXSEG3EI32_V
        | RiscvInstr::VSOXSEG4EI32_V
        | RiscvInstr::VSOXSEG5EI32_V
        | RiscvInstr::VSOXSEG6EI32_V
        | RiscvInstr::VSOXSEG7EI32_V
        | RiscvInstr::VSOXSEG8EI32_V => vector_indexed_ordered_store::<2>,

        RiscvInstr::VSOXEI64_V
        | RiscvInstr::VSOXSEG2EI64_V
        | RiscvInstr::VSOXSEG3EI64_V
        | RiscvInstr::VSOXSEG4EI64_V
        | RiscvInstr::VSOXSEG5EI64_V
        | RiscvInstr::VSOXSEG6EI64_V
        | RiscvInstr::VSOXSEG7EI64_V
        | RiscvInstr::VSOXSEG8EI64_V => vector_indexed_ordered_store::<3>,

        RiscvInstr::VSLIDE1UP_VX => vec_integer_spec_op::<{ vector_spec_instr::SLIDE1UP_VX }>,
        RiscvInstr::VSLIDE1DOWN_VX => vec_integer_spec_op::<{ vector_spec_instr::SLIDE1DOWN_VX }>,
    }
}
