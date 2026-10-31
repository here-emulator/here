#[rustfmt::skip]
#[derive(Default)]
pub struct RVSExtCsrRegfile {
    // S
    // @shadow sstatus:     CSRDataType,
    // @shadow sie:         CSRDataType,
    stvec:          CSRDataType,
    sscratch:       CSRDataType,
    sepc:           CSRDataType,
    scause:         CSRDataType,
    stval:          CSRDataType,
    // @shadow sip:         CSRDataType,
    satp:           CSRDataType,
}