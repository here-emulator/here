pub mod soft_float;
pub mod utils;

pub use utils::BinaryOp;
pub use utils::CmpOp;
pub use utils::FloatPoint;
pub use utils::InFloat;

#[repr(C)]
pub enum Classification {
    NegativeInfinity = 0x1,
    NormalNegative = 0x2,
    SubnormalNegative = 0x4,
    NegativeZero = 0x8,
    PositiveZero = 0x10,
    SubnormalPositive = 0x20,
    NormalPositive = 0x40,
    PositiveInfinity = 0x80,
    SignalingNaN = 0x100,
    QuietNaN = 0x200,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Round {
    NearestTiesToEven,
    TowardPositive,
    TowardNegative,
    TowardZero,
    NearestTiesToAway,
}
