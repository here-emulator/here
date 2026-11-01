use log::error;

use crate::riscv::csr_reg::regfile::{CsrIndex, CsrReadFn, CsrRegfile, CsrWriteFn};

#[macro_use]
pub mod r#macro;

pub mod basic;
pub mod f;
pub mod s;
pub mod v;

pub(self) fn registe_csr(
    regfile: &mut CsrRegfile,
    addr: CsrIndex,
    io_func: (CsrReadFn, CsrWriteFn),
) {
    match addr {
        CsrIndex::Direct(addr) => {
            regfile
                .map
                .insert(adrr, (io_func.0, io_func.1))
                .unwrap_or_else(|| {
                    let mes = format!(
                        "attempted to register two CSRs with the same address{}",
                        addr
                    );
                    error!("{}", mes);
                    panic!("{}", mes)
                });
        }
        _ => todo!(),
    }
}
