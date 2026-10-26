fn get_commit_hash() -> String {
    if let Ok(hash) = std::env::var("GIT_HASH")
        && !hash.is_empty()
    {
        return hash;
    }

    let output = std::process::Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .output();

    match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_owned(),
        _ => String::from("unknown"),
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=GIT_HASH");
    println!("cargo:rustc-env=HERE_COMMIT_HASH={}", get_commit_hash());
}
