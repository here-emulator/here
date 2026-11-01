use log::error;

use crate::riscv::csr_reg::regfile::{
    CsrDataType, CsrIndex, CsrItem, CsrReadFn, CsrRegfile, CsrWriteFn,
};

#[macro_use]
pub mod r#macro;

pub mod basic;
pub mod f;
pub mod s;
pub mod v;

pub use basic::*;
pub use f::*;
pub use s::*;
pub use v::*;

pub(self) fn registe_csr(
    regfile: &mut CsrRegfile,
    addr: CsrIndex,
    io_func: (CsrReadFn, CsrWriteFn),
    default_mask: Option<(&mut CsrItem, CsrDataType)>,
) {
    if let Some((item, mask)) = default_mask {
        item.mask = mask;
    }

    match addr {
        CsrIndex::Direct(addr) => {
            if regfile.map.insert(addr, (io_func.0, io_func.1)).is_some() {
                let mes = format!(
                    "attempted to register two CSRs with the same address {:#x}",
                    addr
                );
                error!("{}", mes);
                panic!("{}", mes);
            }
        }
        _ => todo!(),
    }
}
