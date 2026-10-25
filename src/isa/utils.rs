#[macro_export]
macro_rules! define_instr_enum {
    ($isa_name:ident, $($name:ident),* $(,)?) => {
        #[allow(non_camel_case_types)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub enum $isa_name {
            $($name),*
        }

        impl $isa_name {
            pub fn name(&self) -> String {
                let name = match self {
                    $($isa_name::$name => stringify!($name)),*
                };
                name.to_ascii_lowercase().replace('_', ".")
            }
        }
    }
}

pub struct DecodeMask {
    pub key: u32,
    pub mask: u32,
}

impl DecodeMask {
    pub fn matches(&self, instr: u32) -> bool {
        (instr & self.mask) == self.key
    }
}

pub fn create_decode_mask(pattern: &'static str) -> DecodeMask {
    let mut len = 0;
    let mut key = 0 as u32;
    let mut mask = 0 as u32;

    for ch in pattern.chars() {
        match ch {
            '0' | '1' | '?' | '-' => {
                len += 1;

                key = (key << 1) | (ch == '1') as u32;
                mask = (mask << 1) | (ch == '0' || ch == '1') as u32;
            }
            _ => {
                panic!("unexpected char in pattern {}", ch);
            }
        }
    }

    assert!(len <= 32, "Pattern length exceeds 32 bits");
    assert!(len % 8 == 0, "Pattern length is not a multiple of 8");

    DecodeMask {
        key: key as u32,
        mask: mask as u32,
    }
}
