use xmas_elf::symbol_table::{Entry, Entry32, Entry64};

use crate::{config::arch_config::WordType, ram::Ram, ram_config::BASE_ADDR, utils::BiMap};

pub struct SymTab {
    pub symbols: BiMap<String, u64>,
}

impl SymTab {
    pub fn from(symbols: &[(String, u64)]) -> Self {
        let mut symbol_table = BiMap::<String, u64>::new();
        for (name, addr) in symbols.iter().cloned() {
            symbol_table.insert(name, addr);
        }
        SymTab {
            symbols: symbol_table,
        }
    }

    pub fn func_addr_by_name(&self, name: &str) -> Option<u64> {
        self.symbols.get_by_left(&name.to_string()).cloned()
    }

    pub fn func_name_by_addr(&self, addr: u64) -> Option<&String> {
        self.symbols.get_by_right(&addr)
    }

    pub fn func_name_in_addr_range(&self, addr: u64) -> Option<&String> {
        self.symbols
            .backward
            .range(..=addr)
            .next_back()
            .map(|(_, name)| name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &u64)> {
        self.symbols.iter()
    }
}

pub struct ELFLoader {
    elf_data: Vec<u8>,
}

impl ELFLoader {
    pub fn try_new(elf_data: Vec<u8>) -> Option<ELFLoader> {
        if xmas_elf::ElfFile::new(&elf_data).is_err() {
            return None;
        }
        Some(ELFLoader { elf_data })
    }

    fn elf(&'_ self) -> xmas_elf::ElfFile<'_> {
        xmas_elf::ElfFile::new(&self.elf_data).unwrap()
    }

    pub fn load_to_ram(&self, ram: &mut Ram) {
        let elf = self.elf();
        for ph in elf.program_iter() {
            if ph.get_type().unwrap() == xmas_elf::program::Type::Load {
                let start_addr = ph.virtual_addr() as WordType;
                // let end_addr = ((ph.virtual_addr() + ph.mem_size()) as usize) as WordType;

                ram.insert_section(
                    &elf.input[ph.offset() as usize..(ph.offset() + ph.file_size()) as usize],
                    start_addr - BASE_ADDR,
                );
            }
        }
    }

    pub fn get_section_addr(&self, section_name: &str) -> Option<WordType> {
        let elf = self.elf();
        for sh in elf.section_iter() {
            if let Ok(name) = sh.get_name(&elf) {
                if name == section_name {
                    let addr = sh.address();
                    return Some(addr as WordType);
                }
            }
        }

        None
    }

    fn parse_symtab<T: Entry>(&self, entries: &[T]) -> Option<SymTab> {
        let mut func_table = BiMap::<String, u64>::new();

        for entry in entries {
            // if entry.get_type().unwrap() == xmas_elf::symbol_table::Type::Func {
            let name = entry.get_name(&self.elf()).ok()?.to_string();
            let addr = entry.value();
            func_table.insert(name, addr);
            // }
        }

        Some(SymTab {
            symbols: func_table,
        })
    }

    pub fn get_symbol_table(&self) -> Option<SymTab> {
        let elf = self.elf();
        for sh in elf.section_iter() {
            if let Ok(name) = sh.get_name(&elf)
                && name == ".symtab"
            {
                return match sh.get_data(&elf).ok()? {
                    xmas_elf::sections::SectionData::SymbolTable32(symtab) => {
                        self.parse_symtab::<Entry32>(symtab)
                    }
                    xmas_elf::sections::SectionData::SymbolTable64(symtab) => {
                        self.parse_symtab::<Entry64>(symtab)
                    }
                    _ => None,
                };
            }
        }

        None
    }
}

pub fn load_bin(ram: &mut Ram, raw_data: &[u8]) {
    ram.insert_section(raw_data, 0);
}
