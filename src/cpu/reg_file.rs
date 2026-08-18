use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
};

use crate::config::arch_config::{REG_NAME, REGFILE_CNT, WordType};

#[repr(transparent)]
pub struct RegFile {
    data: [WordType; REGFILE_CNT],
}

impl Index<usize> for RegFile {
    type Output = WordType;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for RegFile {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl Debug for RegFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let byte_len = size_of::<WordType>();
        let hex_width = byte_len * 2; // 每字节2位16进制

        writeln!(f, "reg_file {{")?;
        for (i, val) in self.data.iter().enumerate() {
            // 每行8个
            if i % 8 == 0 {
                write!(f, "  ")?; // 缩进
            }

            write!(
                f,
                "{:>6}: 0x{:0width$x}  ",
                REG_NAME[i],
                val,
                width = hex_width
            )?;

            if i % 8 == 7 {
                writeln!(f)?;
            }
        }

        // 若最后一行不足8个，手动换行
        if self.data.len() % 8 != 0 {
            writeln!(f)?;
        }

        write!(f, "}}")
    }
}

impl RegFile {
    pub fn new() -> Self {
        Self {
            data: [0; REGFILE_CNT],
        }
    }

    pub fn read(&self, id1: u8, id2: u8) -> (WordType, WordType) {
        (self.data[id1 as usize], self.data[id2 as usize])
    }

    #[cfg(any(feature = "riscv64", feature = "riscv32"))]
    /// id == 0 will be ignored, if an instruction do not need to WriteBack, set id = 0.
    pub fn write(&mut self, id: u8, data: WordType) {
        if id == 0u8 {
            return;
        }

        self.data[id as usize] = data
    }
}
