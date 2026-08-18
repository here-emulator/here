pub mod x86;

trait Emitable {
    fn emit_to(self, buf: &mut Vec<u8>);
}

impl Emitable for u8 {
    fn emit_to(self, buf: &mut Vec<u8>) {
        buf.push(self);
    }
}

impl Emitable for u16 {
    fn emit_to(self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_le_bytes());
    }
}

impl Emitable for u32 {
    fn emit_to(self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_le_bytes());
    }
}

impl Emitable for u64 {
    fn emit_to(self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_le_bytes());
    }
}

impl Emitable for &[u8] {
    fn emit_to(self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(self);
    }
}

impl<const N: usize> Emitable for [u8; N] {
    fn emit_to(self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self);
    }
}

macro_rules! define_emitable_signed {
    ($t:ty) => {
        impl Emitable for $t {
            fn emit_to(self, buf: &mut Vec<u8>) {
                self.cast_unsigned().emit_to(buf);
            }
        }
    };
}

define_emitable_signed! { i8 }
define_emitable_signed! { i16 }
define_emitable_signed! { i32 }
define_emitable_signed! { i64 }
