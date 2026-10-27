pub mod ram_config {
    use crate::config::arch_config::WordType;
    pub const BASE_ADDR: WordType = 0x8000_0000;
    pub const DEFAULT_PC_VALUE: WordType = BASE_ADDR;

    pub const SIZE: usize = 512 << 20;
}

pub mod arch_config {
    #[cfg(feature = "riscv32")]
    pub type WordType = u32;
    #[cfg(feature = "riscv64")]
    pub type WordType = u64;

    #[cfg(feature = "riscv32")]
    pub type SignedWordType = i32;
    #[cfg(feature = "riscv64")]
    pub type SignedWordType = i64;

    pub const REGFILE_CNT: usize = 32;
    pub const REG_NAME: [&str; REGFILE_CNT] = [
        "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0/fp", "s1", "a0", "a1", "a2", "a3",
        "a4", "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3",
        "t4", "t5", "t6",
    ];

    pub const FLOAT_REGFILE_CNT: usize = 32;
    pub const FLOAT_REG_NAME: [&str; FLOAT_REGFILE_CNT] = [
        "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "fs0", "fs1", "fa0", "fa1", "fa2",
        "fa3", "fa4", "fa5", "fa6", "fa7", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7", "fs8", "fs9",
        "fs10", "fs11", "ft8", "ft9", "ft10", "ft11",
    ];

    pub const VECTOR_REGFILE_CNT: usize = 32;
    pub const VECTOR_REG_NAME: [&str; VECTOR_REGFILE_CNT] = [
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
        "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v27", "v28", "v29", "v30", "v31",
    ];

    pub const XLEN: usize = size_of::<WordType>() << 3;
}
