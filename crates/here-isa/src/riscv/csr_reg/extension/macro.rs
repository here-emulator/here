/// Generate bitfield getters, setters, and readonly mask helpers for a CSR register.
macro_rules! gen_csr_bitfield {
    (
        $name:ident, $sub_regfile:ident, $field:ident, [
            $( $bit:expr, $len:expr, $fname:ident $(, $ro:ident)? );* $(;)?
        ]
    ) => {
        impl $name {
            #[allow(dead_code)]
            pub const SHADOW_MASK: CsrDataType = 0 $( | Self::${concat($field, _, $fname, _mask)} )*;

            #[allow(dead_code)]
            pub const DEFAULT_WRITABLE_MASK: CsrDataType =
                0 $( | Self::${concat($field, _, $fname, _write_mask)} )*;

            $(
                // Offset
                #[allow(non_upper_case_globals)]
                const ${concat($field, _, $fname, _offset)}: usize = if $bit >= 0 {
                    ($bit as isize) as usize
                } else {
                    ((XLEN as isize) + ($bit as isize)) as usize
                };

                // Mask
                #[allow(non_upper_case_globals)]
                const ${concat($field, _, $fname, _mask)}: CsrDataType =
                    BIT_ONES_ARRAY[$len as usize] << Self::${concat($field, _, $fname, _offset)};

                // Write Mask
                #[allow(non_upper_case_globals)]
                const ${concat($field, _, $fname, _write_mask)}: CsrDataType =
                    gen_csr_bitfield!(@write_mask $field, $fname $(, $ro)?);

                // Get value
                #[inline]
                pub fn ${concat(get_, $fname)}(regfile: &CsrRegfile) -> CsrDataType {
                    (Self::read(regfile) & Self::${concat($field, _, $fname, _mask)})
                        >> Self::${concat($field, _, $fname, _offset)}
                }

                // Set value
                #[inline]
                pub fn ${concat(set_, $fname)}(
                    regfile: &mut CsrRegfile,
                    ${concat(new_, $fname)}: CsrDataType,
                ) {
                    let mut data = Self::read(regfile);
                    data &= !Self::${concat($field, _, $fname, _mask)};
                    let ${concat(new_, $fname)} = (${concat(new_, $fname)}
                        << Self::${concat($field, _, $fname, _offset)})
                        & Self::${concat($field, _, $fname, _mask)};
                    data |= ${concat(new_, $fname)};
                    Self::write(regfile, data);
                }

                // Set value directly without validation
                #[inline]
                pub fn ${concat(set_, $fname, _directly)}(
                    regfile: &mut CsrRegfile,
                    ${concat(new_, $fname)}: CsrDataType,
                ) {
                    let mut data = Self::read(regfile);
                    data &= !Self::${concat($field, _, $fname, _mask)};
                    let ${concat(new_, $fname)} = (${concat(new_, $fname)}
                        << Self::${concat($field, _, $fname, _offset)})
                        & Self::${concat($field, _, $fname, _mask)};
                    data |= ${concat(new_, $fname)};
                    Self::write_directly(regfile, data);
                }
                // Set Read Only
                #[inline]
                pub fn ${concat(set_, $fname, _readonly)}(regfile: &mut CsrRegfile) {
                    regfile.$sub_regfile.$field.mask &= !Self::${concat($field, _, $fname, _mask)};
                }
            )*
        }
    };

    (@write_mask $field:ident, $fname:ident, RO) => {
        0
    };

    (@write_mask $field:ident, $fname:ident) => {
        Self::${concat($field, _, $fname, _mask)}
    };
}

/// Generate a single CSR register.
macro_rules! gen_csr_reg {
    (
        $name:ident, $extension:expr, $sub_regfile:ident, $field:ident, $index:expr,
        $(@$validate_tag:ident)? [
            $( $bit:expr, $len:expr, $fname:ident $(, $ro:ident)? );* $(;)?
        ]
    ) => {
        pub struct $name {}

        gen_csr_reg!(@impl_validator $name, $sub_regfile, $field $(, @$validate_tag)?);

        impl NamedCsr for $name {
            const INDEX: CsrIndex = $index;
            const EXTENSION: Extension = $extension;

            fn register(regfile: &mut CsrRegfile) {
                let default_mask = (
                    &mut regfile.$sub_regfile.$field as *mut CsrItem,
                    Self::DEFAULT_WRITABLE_MASK,
                );
                registe_csr(
                    regfile,
                    Self::INDEX,
                    (Self::read, Self::write),
                    Some((unsafe { &mut *default_mask.0 }, default_mask.1)),
                );
            }

            #[inline(always)]
            fn read(regfile: &CsrRegfile) -> CsrDataType {
                regfile.$sub_regfile.$field.data
            }

            #[inline(always)]
            fn write(regfile: &mut CsrRegfile, new_value: CsrDataType) {
                let validated_value = Self::validate(regfile, new_value);
                let mask = regfile.$sub_regfile.$field.mask;
                let old = regfile.$sub_regfile.$field.data;
                regfile.$sub_regfile.$field.data = (old & !mask) | (validated_value & mask);
            }

            #[inline(always)]
            fn write_directly(regfile: &mut CsrRegfile, new_value: CsrDataType) {
                regfile.$sub_regfile.$field.data = new_value;
            }
        }

        gen_csr_bitfield!(
            $name, $sub_regfile, $field, [
                $( $bit, $len, $fname $(, $ro)? );*
            ]
        );
    };

    (@impl_validator $name:ident, $sub_regfile:ident, $field:ident, @validate_custom) => {
        impl CsrValidator<$name> for $name {
            fn validate(regfile: &mut CsrRegfile, new_value: CsrDataType) -> CsrDataType {
                let value = regfile.$sub_regfile.$field.mask & new_value;
                Self::validate_custom(regfile, value)
            }
        }
    };

    (@impl_validator $name:ident, $sub_regfile:ident, $field:ident) => {
        impl CsrValidator<$name> for $name {
            fn validate(regfile: &mut CsrRegfile, new_value: CsrDataType) -> CsrDataType {
                regfile.$sub_regfile.$field.mask & new_value
            }
        }
    };
}

/// Generate a sub-extension CSR Regfile registration and all its NamedCsr types.
macro_rules! gen_csr_regfile {
    (
        $regfile_type:ty {
            $(
                $name:ident, $extension:expr, $sub_regfile:ident, $field:ident, $index:expr,
                $(@$validate_tag:ident)? [
                    $( $bit:expr, $len:expr, $fname:ident $(, $ro:ident)? );* $(;)?
                ]
            );* $(;)?
        }
    ) => {
        impl $regfile_type {
            fn macro_register(regfile: &mut CsrRegfile) {
                $(
                    $name::register(regfile);
                )*
            }
        }

        $(
            gen_csr_reg!(
                $name, $extension, $sub_regfile, $field, $index,
                $(@$validate_tag)? [
                    $( $bit, $len, $fname $(, $ro)? );*
                ]
            );
        )*
    };
}
