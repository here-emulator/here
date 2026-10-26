use crate::{config::arch_config::WordType, utils::make_mask};

#[derive(Clone, Copy, Debug)]
pub(super) struct ReadValidator {
    pub(super) target_index: WordType,
    pub(super) view_mask: WordType,
}

impl ReadValidator {
    pub(super) fn new(target_index: WordType, mask: WordType) -> Self {
        Self {
            target_index,
            view_mask: mask,
        }
    }

    pub(super) fn apply(&self, value: WordType) -> WordType {
        value & self.view_mask
    }

    pub(super) const fn combine(&self, other: WordType) -> ReadValidator {
        ReadValidator {
            target_index: self.target_index,
            view_mask: self.view_mask | other,
        }
    }
}

pub(super) fn gen_shadow_read_mask<const L: usize, const R: usize>() -> WordType {
    make_mask(L, R)
}

macro_rules! combine_shadow_read_ops {
    ($target_index:expr, $($left: expr, $right: expr),* $(,)?) => {
        {
            #[allow(unused_mut)]
            let mut combined = ReadValidator::new($target_index, 0);
            $(
                combined = combined.combine(gen_shadow_read_mask::<{$left}, {$right}>());
            )*
            combined
        }
    };
}
