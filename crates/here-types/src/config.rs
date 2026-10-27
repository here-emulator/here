pub mod ram_config {
    use crate::config::arch_config::WordType;
    pub const BASE_ADDR: WordType = 0x8000_0000;
    pub const DEFAULT_PC_VALUE: WordType = BASE_ADDR;

    pub const SIZE: usize = 512 << 20;
}

pub mod arch_config {
    use crate::gen_reg_name_list;

    macro_rules! arch_config {
        (
            $(
                @item
                $feature:literal => {
                    word: $word:ty,
                    signed_word: $signed_word:ty,
                    reg_name: $reg_name: expr,
                    float_reg_name: $float_reg_name: expr,
                    vector_reg_name: $vector_reg_name: expr,
                }
            ),* $(,)?
        ) => {
            $(
                #[cfg(feature = $feature)]
                pub type WordType = $word;

                #[cfg(feature = $feature)]
                pub type SignedWordType = $signed_word;

                #[cfg(feature = $feature)]
                pub const REGFILE_CNT: usize = $reg_name.len();

                #[cfg(feature = $feature)]
                pub const REG_NAME: [&str; REGFILE_CNT] = $reg_name;

                #[cfg(feature = $feature)]
                pub const FLOAT_REGFILE_CNT: usize = $float_reg_name.len();

                #[cfg(feature = $feature)]
                pub const FLOAT_REG_NAME: [&str; FLOAT_REGFILE_CNT] = $float_reg_name;

                #[cfg(feature = $feature)]
                pub const VECTOR_REGFILE_CNT: usize = $vector_reg_name.len();

                #[cfg(feature = $feature)]
                pub const VECTOR_REG_NAME: [&str; VECTOR_REGFILE_CNT] = $vector_reg_name;
            )*
        };
    }
    pub const XLEN: usize = size_of::<WordType>() << 3;

    arch_config! {
        @item "riscv32" => {
            word: u32,
            signed_word: i32,
            reg_name: gen_reg_name_list!(   "zero";     "ra";           "sp";       "gp";
                                            "tp";       "t", 0, 2;      "s0/fp";    "s1";
                                            "a", 0, 7;  "s", 2, 11;     "t", 3, 6),
            float_reg_name: gen_reg_name_list!("ft", 0, 7; "fs", 0, 1; "fa", 0, 7; "fs", 2, 11; "ft", 8, 11),
            vector_reg_name: gen_reg_name_list!("v", 0, 31),
        },
        @item "riscv64" => {
            word: u64,
            signed_word: i64,
            reg_name: gen_reg_name_list!(   "zero";     "ra";   "sp";   "gp";
                                            "tp";       "t", 0, 2;      "s0/fp";    "s1";
                                            "a", 0, 7;  "s", 2, 11;     "t", 3, 6),
            float_reg_name: gen_reg_name_list!("ft", 0, 7; "fs", 0, 1; "fa", 0, 7; "fs", 2, 11; "ft", 8, 11),
            vector_reg_name: gen_reg_name_list!("v", 0, 31),
        }
    }
}
