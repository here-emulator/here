#[rustfmt::skip]
#[derive(Default)]
pub struct RVVExtCsrRegfile {
    vstart:         CSRDataType,
    vcsr:           CSRDataType,
    // @shadow cycle:          CSRDataType,
    // @shadow instret:        CSRDataType,
    vl:             CSRDataType,
    vtype:          CSRDataType,
    vlenb:          CSRDataType,
}
