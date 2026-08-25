---
name: bug-report
description: >-
  Instructs the agent on how to reproduce a reported bug, collect emulator diagnostics and system metadata,
  and submit a structured bug report to GitHub using the `gh` CLI (with fallback to manual blank issue submission)
  for here-emulator/here, including mandatory user preview and confirmation.
---

# Bug Report Submission Workflow

This skill guides the agent through investigating, reproducing, and submitting high-quality bug reports for the RISC-V emulator (`HERE`) to the repository [here-emulator/here](https://github.com/here-emulator/here/issues).

---

## Workflow Overview

```
1. Reproduce & Collect Diagnostics
   ├── Run reproduction commands (cargo run / cargo test)
   ├── Collect environment info (OS, Rust version, Git commit hash)
   └── Capture error logs and traces (--loglevel=trace, RUST_BACKTRACE=1)
          │
2. GitHub CLI (gh) Check
   ├── Found & Authenticated? ──> Proceed to draft
   └── Not found?
       ├── Detect OS distribution & ask user before installing
       └── If declined/failed ──> Prepare manual fallback
          │
3. Draft Bug Report (Conforming to bug_report.yml template)
          │
4. Mandatory User Preview & Confirmation (ALWAYS ask before submitting)
          │
5. Submit
   ├── Method A (gh CLI): gh issue create --repo here-emulator/here ...
   └── Method B (Manual): Guide user to https://github.com/here-emulator/here/issues
```

---

## Step 1: Local Reproduction & Diagnostic Collection

Before drafting an issue, attempt to reproduce the bug locally to collect precise logs:

1. **Analyze User Input**: Extract the command line arguments, target ELF/binary, ISA flags, or test case mentioned by the user.
2. **Attempt Reproduction**:
   - Run the emulator command with debug logging:
     ```sh
     cargo run -- <binary_path> --loglevel=trace
     ```
   - If a panic or crash occurs, rerun with backtrace:
     ```sh
     RUST_BACKTRACE=1 cargo run -- <binary_path> [args]
     ```
   - If it is a test failure:
     ```sh
     cargo test <test_name> -- --nocapture
     ```
3. **Gather System & Toolchain Information**:
   - **Git Commit Hash**: `git rev-parse --short HEAD` (or release version)
   - **OS & Architecture**: `uname -s`, `uname -m`, or inspect `/etc/os-release`
   - **Rust Version**: `rustc --version`
   - **Cargo Version**: `cargo --version`

---

## Step 2: GitHub CLI (`gh`) Availability & Installation

Check whether the GitHub CLI is available on the user's system:

```sh
command -v gh && gh auth status
```

### If `gh` is NOT installed:
1. Detect the operating system and package manager.
2. **Ask the user** if they would like to install `gh` via their package manager.
3. **If user agrees**: Attempt installation and verify with `gh auth status` (guide user to run `gh auth login` if unauthenticated).
4. **If user declines or installation fails**: Smoothly transition to **Manual Submission Mode** (Step 5 - Method B).

---

## Step 3: Format the Bug Report

Format the report according to `.github/ISSUE_TEMPLATE/bug_report.yml`:

```markdown
### Is there an existing issue for this?
- [x] I have searched the existing issues

### Platform
<Platform, e.g., Linux (x86_64), macOS (Apple Silicon - aarch64), Windows (x86_64)>

### Version
<Git commit hash, e.g. 78bd5fd or release tag v0.1.0>

### Describe the bug
#### Bug Description
<Clear and concise description of the unexpected behavior>

#### Steps to Reproduce
1. Command: `cargo run -- <binary> [options]`
2. Target binary / workload: `<file path or description>`
3. Flags / features used: `<e.g. --device=virtio-block:...>`

#### Expected Behavior
<What should have happened instead>

### Related Logs
```shell
<Paste captured emulator logs, panic backtrace, or terminal output here>
```

### Additional context
- Rust toolchain: `<rustc --version output>`
- Host OS: `<OS and kernel info>`
- Guest OS / SBI: `<e.g., Linux 6.18.2 / OpenSBI if applicable>`
```

---

## Step 4: Mandatory User Preview & Confirmation

> [!IMPORTANT]
> **Always preview the entire bug report to the user before submitting.**
> Never submit an issue via `gh` or finish without first showing the generated Markdown and asking for explicit confirmation.

Show the user:
1. **Title**: e.g., `[Bug]: <Concise Summary>`
2. **Target Repository**: `here-emulator/here`
3. **Report Body Preview**: The complete Markdown formatted text.
4. **Prompt for Confirmation**: Ask the user: *"Please review the bug report above. Would you like me to submit this issue now?"*

---

## Step 5: Submission

### Method A: Automated Submission via GitHub CLI (`gh`)
Once the user gives confirmation:
1. Create the issue using `gh`:
   ```sh
   gh issue create \
     --repo here-emulator/here \
     --title "[Bug]: <Issue Title>" \
     --body "<Formatted Markdown Body>" \
     --label "bug"
   ```
2. Output the resulting Issue URL to the user.

### Method B: Manual Submission via Blank Issue (Fallback)
If `gh` is unavailable or user preferred manual submission:
1. Provide the finalized Markdown text inside a code block for easy copying.
2. Instruct the user to:
   1. Open [https://github.com/here-emulator/here/issues](https://github.com/here-emulator/here/issues) in their browser.
   2. Click **"New issue"**.
   3. Choose **"blank_issues"** (or create a blank issue).
   4. Set Title to `[Bug]: <Issue Title>` and paste the copied Markdown into the issue body.
   5. Click **"Submit new issue"**.
