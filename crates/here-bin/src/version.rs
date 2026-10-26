const fn has_dash(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'-' {
            return true;
        }
        i += 1;
    }
    false
}

// it is a macro only for test reason, otherwise it's easy to rewrite into a const fn
macro_rules! compute_version {
    ($pkg:expr, $is_release:expr, $hash:expr) => {
        if $is_release {
            if has_dash($pkg) {
                concat!($pkg, " (", $hash, ")")
            } else {
                $pkg
            }
        } else {
            concat!($pkg, " WIP (", $hash, ")")
        }
    };
}

pub const VERSION: &str = compute_version!(
    env!("CARGO_PKG_VERSION"),
    option_env!("HERE_RELEASE").is_some(),
    env!("HERE_COMMIT_HASH")
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_release() {
        const V: &str = compute_version!("0.2.0", true, "a1b2c3d4");
        assert_eq!(V, "0.2.0");
    }

    #[test]
    fn prerelease() {
        const V: &str = compute_version!("0.2.0-alpha", true, "a1b2c3d4");
        assert_eq!(V, "0.2.0-alpha (a1b2c3d4)");
    }

    #[test]
    fn dev() {
        const V: &str = compute_version!("0.2.0", false, "a1b2c3d4");
        assert_eq!(V, "0.2.0 WIP (a1b2c3d4)");
    }

    #[test]
    fn dev_prerelease() {
        const V: &str = compute_version!("0.2.0-alpha", false, "a1b2c3d4");
        assert_eq!(V, "0.2.0-alpha WIP (a1b2c3d4)");
    }
}
