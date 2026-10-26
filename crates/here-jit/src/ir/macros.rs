/// Helper for writing a sequence of [`ir::IRBuilder`] operations.
///
/// # Example
///
/// ```ignore
/// # use here_jit::make_ir;
/// # use here_jit::ir::IRBuilder;
/// # let mut builder = IRBuilder::new();
/// let (v1, v2) = make_ir!(builder, [
///     v0 = const 1024 * 2;
///     v1 = get_reg 1;
///     v2 = compute Add, v0, v1;
///     set_reg 2, v2;
/// ], [v1, v2]);
/// let block = builder.build_jmp(0x1000u64.into());
/// ```
///
/// Calls use named attributes instead of positional booleans. `trap` marks calls
/// that can exit through `longjmp`.
/// A left-hand side requests the helper return value:
///
/// ```text
/// value = call [trap] helper, [address];
/// call [] helper, [address, value];
/// ```
#[macro_export]
macro_rules! make_ir {
    ($builder:expr, [$($statements:tt)*], [$($returns:expr),* $(,)?]) => {{
        let __make_ir_builder = &mut $builder;
        $crate::make_ir!(@statements __make_ir_builder; [$($returns),*]; $($statements)*)
    }};

    (@statements $builder:ident; [$($returns:expr),*];) => {
        $crate::make_ir!(@returns [$($returns),*])
    };

    (@statements $builder:ident; [$($returns:expr),*];
        $name:ident = get_reg $guest:expr;
        $($rest:tt)*) => {
        {
            let $name = $builder.get_reg(($guest).into());
            $crate::make_ir!(@statements $builder; [$($returns),*]; $($rest)*)
        }
    };

    (@statements $builder:ident; [$($returns:expr),*];
        $name:ident = const $imm:expr;
        $($rest:tt)*) => {
        {
            let $name = $builder.const_value($imm);
            $crate::make_ir!(@statements $builder; [$($returns),*]; $($rest)*)
        }
    };

    (@statements $builder:ident; [$($returns:expr),*];
        $name:ident = compute $kind:expr, $lhs:expr, $rhs:expr;
        $($rest:tt)*) => {
        {
            use $crate::ir::ComputeKind::*;
            let $name = $builder.compute(
                $kind,
                ($lhs).into(),
                ($rhs).into(),
            );
            $crate::make_ir!(@statements $builder; [$($returns),*]; $($rest)*)
        }
    };

    (@statements $builder:ident; [$($returns:expr),*];
        set_reg $guest:expr, $value:expr;
        $($rest:tt)*) => {
        {
            $builder.set_reg(($guest).into(), ($value).into());
            $crate::make_ir!(@statements $builder; [$($returns),*]; $($rest)*)
        }
    };

    (@statements $builder:ident; [$($returns:expr),*];
        $name:ident = call [$($option:ident),* $(,)?] $func_ptr:expr, [$($arg:expr),* $(,)?];
        $($rest:tt)*) => {
        {
            let __make_ir_call_args =
                vec![$(($arg).into()),*];
            let __make_ir_call_options =
                $crate::make_ir!(@call_options [$($option),*]);
            let $name = $builder.call_ret(
                $func_ptr,
                __make_ir_call_args,
                __make_ir_call_options,
            );
            $crate::make_ir!(@statements $builder; [$($returns),*]; $($rest)*)
        }
    };

    (@statements $builder:ident; [$($returns:expr),*];
        call [$($option:ident),* $(,)?] $func_ptr:expr, [$($arg:expr),* $(,)?];
        $($rest:tt)*) => {
        {
            let __make_ir_call_args =
                vec![$(($arg).into()),*];
            let __make_ir_call_options =
                $crate::make_ir!(@call_options [$($option),*]);
            $builder.call(
                $func_ptr,
                __make_ir_call_args,
                __make_ir_call_options,
            );
            $crate::make_ir!(@statements $builder; [$($returns),*]; $($rest)*)
        }
    };

    (@call_options [$($option:ident),* $(,)?]) => {{
        let options = $crate::ir::CallOptions::default();
        $(
            let options = $crate::make_ir!(@call_option options, $option);
        )*
        options
    }};


    (@call_option $options:expr, trap) => {
        ($options).trapping()
    };

    (@returns []) => { () };
    (@returns [$value:expr $(,)?]) => { $value };
    (@returns [$first:expr, $($rest:expr),+ $(,)?]) => {
        ($first, $($rest),+)
    };
}

#[cfg(test)]
mod tests {
    use crate::ir::IRBuilder;
    use crate::ir::IRInst;
    use crate::ir::Value;

    #[test]
    fn call_attributes_and_return_binding() {
        let mut builder = IRBuilder::default();
        let value = make_ir!(builder, [
            value = call [trap] 0usize, [1u64];
            call [] 1usize, [value];
        ], [value]);
        let block = builder.build_jmp(0u64.into());

        let calls = block
            .insts()
            .iter()
            .filter_map(|inst| match inst {
                IRInst::Call { can_trap, ret, .. } => Some((*can_trap, *ret)),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(calls.len(), 2);
        let Value::VReg(value) = value else {
            unreachable!();
        };
        assert_eq!(calls[0], (true, Some(value)));
        assert_eq!(calls[1], (false, None));
    }
}
