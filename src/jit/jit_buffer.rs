use region::{Allocation, Protection};

use super::runtime::JitFunction;

pub const JIT_BUFFER_SIZE: usize = 2 * 1024 * 1024 * 1024; // 2 GiB

/// Storage for generated code in a single 2 GiB virtual address space allocation.
pub(super) struct JitBuffer {
    alloc: Allocation,
    cursor: usize,
    peak_used_bytes: usize,
    #[cfg(target_os = "linux")]
    perf_map: Option<PerfMap>,
}

impl JitBuffer {
    pub fn new() -> Self {
        let alloc = region::alloc(JIT_BUFFER_SIZE, Protection::READ_WRITE_EXECUTE)
            .expect("failed to allocate 2 GiB JIT code buffer");

        #[cfg(target_os = "linux")]
        let mut perf_map = PerfMap::new();
        #[cfg(target_os = "linux")]
        if let Some(ref mut map) = perf_map {
            map.register_chunk(alloc.as_ptr::<u8>() as usize, alloc.len());
        }

        Self {
            alloc,
            cursor: 0,
            peak_used_bytes: 0,
            #[cfg(target_os = "linux")]
            perf_map,
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.alloc.len()
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        self.capacity() - self.cursor
    }

    pub unsafe fn emit_code(&mut self, code: &[u8]) -> JitFunction {
        let len = code.len();
        assert!(len != 0, "cannot emit an empty JIT function");
        assert!(
            self.remaining() >= len,
            "JIT buffer exhausted (exceeded 2 GiB limit)"
        );

        let offset = self.cursor;
        self.cursor += len;

        let function = unsafe {
            let dst = self.alloc.as_mut_ptr::<u8>().add(offset);
            std::ptr::copy_nonoverlapping(code.as_ptr(), dst, len);
            JitFunction::from_ptr(dst)
        };
        self.peak_used_bytes = self.peak_used_bytes.max(self.cursor);

        // FIXME: call clear cache here

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
        let mut buffer = JitBuffer::new();
        let first = unsafe { buffer.emit_code(&[0xc3]) }.address();

        buffer.reset();

        let second = unsafe { buffer.emit_code(&[0xc3]) }.address();
        assert_eq!(second, first);
    }

    #[test]
    fn keeps_the_peak_after_reset() {
        let mut buffer = JitBuffer::new();
        let code = [0xcc; 1024];

        unsafe { buffer.emit_code(&code) };
        buffer.reset();
        unsafe { buffer.emit_code(&[0xcc]) };

        assert_eq!(buffer.latest_used_bytes(), 1);
        assert_eq!(buffer.peak_used_bytes(), 1024);
    }
}
