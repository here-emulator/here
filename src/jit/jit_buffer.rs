use region::{Allocation, Protection};

use super::runtime::JitFunction;

pub const JIT_BUFFER_SIZE: usize = 2 * 1024 * 1024 * 1024;

/// Storage for generated code in a single virtual address space allocation.
pub(super) struct JitBuffer {
    alloc: Allocation,
    cursor: usize,
    peak_used_bytes: usize,
}

impl JitBuffer {
    pub fn with_capacity(capacity: usize) -> Self {
        // TODO(fixme): maybe it cannot work on MacOS or whatever, replace with dual mapping
        let alloc = region::alloc(capacity, Protection::READ_WRITE_EXECUTE)
            .expect("failed to allocate JIT buffer");

        #[cfg(target_os = "linux")]
        if let Some(mut map) = PerfMap::new() {
            map.register_chunk(alloc.as_ptr::<u8>() as usize, alloc.len());
        }

        Self {
            alloc,
            cursor: 0,
            peak_used_bytes: 0,
        }
    }

    pub fn new() -> Self {
        Self::with_capacity(JIT_BUFFER_SIZE)
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.alloc.len()
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        self.capacity() - self.cursor
    }

    // TODO: return Result or Option when we fail.
    pub unsafe fn emit_code(&mut self, code: &[u8]) -> JitFunction {
        let len = code.len();
        assert!(len != 0, "cannot emit an empty JIT function");
        assert!(self.remaining() >= len, "JIT buffer exhausted");

        let offset = self.cursor;
        self.cursor += len;

        let function = unsafe {
            let dst = self.alloc.as_mut_ptr::<u8>().add(offset);
            std::ptr::copy_nonoverlapping(code.as_ptr(), dst, len);
            JitFunction::from_ptr(dst)
        };
        self.peak_used_bytes = self.peak_used_bytes.max(self.cursor);

        // TODO(fixme): call clear cache here

        function
    }

    #[inline]
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    #[inline]
    pub fn latest_allocated_bytes(&self) -> usize {
        self.alloc.len()
    }

    #[inline]
    pub fn latest_used_bytes(&self) -> usize {
        self.cursor
    }

    #[inline]
    pub fn peak_used_bytes(&self) -> usize {
        self.peak_used_bytes
    }
}

#[cfg(target_os = "linux")]
struct PerfMap {
    file: std::fs::File,
}

#[cfg(target_os = "linux")]
impl PerfMap {
    fn new() -> Option<Self> {
        let pid = std::process::id();
        let path = format!("/tmp/perf-{}.map", pid);
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()?;
        Some(Self { file })
    }

    fn register_chunk(&mut self, start: usize, size: usize) {
        use std::io::Write;
        let _ = writeln!(self.file, "{:x} {:x} jit_code", start, size);
        let _ = self.file.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_the_initial_fixed_size_buffer() {
        let buffer = JitBuffer::new();

        assert_eq!(buffer.latest_allocated_bytes(), JIT_BUFFER_SIZE);
        assert_eq!(buffer.latest_used_bytes(), 0);
    }

    #[test]
    fn reset_reuses_the_code_buffer() {
        let mut buffer = JitBuffer::with_capacity(4096);
        let first = unsafe { buffer.emit_code(&[0xc3]) }.address();

        buffer.reset();

        let second = unsafe { buffer.emit_code(&[0xc3]) }.address();
        assert_eq!(second, first);
    }

    #[test]
    fn keeps_the_peak_after_reset() {
        let mut buffer = JitBuffer::with_capacity(4096);
        let code = [0xcc; 1024];

        unsafe { buffer.emit_code(&code) };
        buffer.reset();
        unsafe { buffer.emit_code(&[0xcc]) };

        assert_eq!(buffer.latest_used_bytes(), 1);
        assert_eq!(buffer.peak_used_bytes(), 1024);
    }
}
