use region::{Allocation, Protection};

use super::jit_function::JitFunction;

const JIT_BUFFER_SIZE: usize = 2 * 1024 * 1024;

struct JitChunk {
    alloc: Allocation,
    cursor: usize,
}

impl JitChunk {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            alloc: region::alloc(capacity, Protection::READ_WRITE_EXECUTE)
                .expect("failed to allocate JIT code buffer"),
            cursor: 0,
        }
    }

    fn capacity(&self) -> usize {
        self.alloc.len()
    }

    fn remaining(&self) -> usize {
        self.capacity() - self.cursor
    }
}

/// Storage for generated code.
pub(super) struct JitBuffer {
    chunks: Vec<JitChunk>,
    current_chunk: usize,
    used_bytes: usize,
    peak_used_bytes: usize,
}

impl Drop for JitBuffer {
    fn drop(&mut self) {
        log::info!("JIT buffer statistics:");
        log::info!("  chunks: {}", self.chunk_count());
        log::info!("  allocated: {:.2} MB", megabytes(self.allocated_bytes()));
        log::info!("  used: {:.2} MB", megabytes(self.used_bytes()));
        log::info!("  peak used: {:.2} MB", megabytes(self.peak_used_bytes()));
    }
}

impl JitBuffer {
    pub fn new() -> Self {
        let mut buffer = Self {
            chunks: Vec::new(),
            current_chunk: 0,
            used_bytes: 0,
            peak_used_bytes: 0,
        };
        buffer.allocate_chunk();
        buffer
    }

    fn allocate_chunk(&mut self) {
        self.chunks.push(JitChunk::with_capacity(JIT_BUFFER_SIZE));
    }

    pub unsafe fn emit_code(&mut self, code: &[u8]) -> JitFunction {
        let len = code.len();
        assert!(len != 0, "cannot emit an empty JIT function");
        assert!(
            len <= JIT_BUFFER_SIZE,
            "JIT function exceeds the fixed 2 MiB code buffer"
        );

        while self.chunks[self.current_chunk].remaining() < len {
            self.current_chunk += 1;
            if self.current_chunk == self.chunks.len() {
                self.allocate_chunk();
            }
        }

        let chunk = &mut self.chunks[self.current_chunk];
        let offset = chunk.cursor;
        chunk.cursor += len;

        let function = unsafe {
            let dst = chunk.alloc.as_mut_ptr::<u8>().add(offset);
            std::ptr::copy_nonoverlapping(code.as_ptr(), dst, len);
            JitFunction::from_ptr(dst)
        };
        self.used_bytes += len;
        self.peak_used_bytes = self.peak_used_bytes.max(self.used_bytes());
        function
    }

    pub fn reset(&mut self) {
        self.current_chunk = 0;
        self.used_bytes = 0;
        for chunk in &mut self.chunks {
            chunk.cursor = 0;
        }
    }

    /// Total executable memory currently reserved from the host.
    pub fn allocated_bytes(&self) -> usize {
        self.chunks.iter().map(JitChunk::capacity).sum()
    }

    /// Bytes containing generated code since the last reset.
    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    /// Maximum generated-code footprint since this buffer was created.
    pub fn peak_used_bytes(&self) -> usize {
        self.peak_used_bytes
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

fn megabytes(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_the_initial_fixed_size_chunk() {
        let buffer = JitBuffer::new();

        assert_eq!(buffer.chunk_count(), 1);
        assert_eq!(buffer.allocated_bytes(), JIT_BUFFER_SIZE);
        assert_eq!(buffer.used_bytes(), 0);
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
    fn allocates_additional_chunks_when_the_current_chunk_is_full() {
        let mut buffer = JitBuffer::new();
        let first = vec![0xcc; JIT_BUFFER_SIZE];
        let second = [0xc3];

        let first_address = unsafe { buffer.emit_code(&first) }.address();
        let second_address = unsafe { buffer.emit_code(&second) }.address();

        assert_eq!(buffer.chunk_count(), 2);
        assert_eq!(buffer.used_bytes(), JIT_BUFFER_SIZE + second.len());
        assert_eq!(buffer.allocated_bytes(), JIT_BUFFER_SIZE * 2);
        assert_ne!(second_address, first_address);
    }

    #[test]
    fn keeps_the_peak_after_reset() {
        let mut buffer = JitBuffer::new();
        let code = vec![0xcc; JIT_BUFFER_SIZE];

        unsafe { buffer.emit_code(&code) };
        buffer.reset();
        unsafe { buffer.emit_code(&[0xcc]) };

        assert_eq!(buffer.used_bytes(), 1);
        assert_eq!(buffer.peak_used_bytes(), JIT_BUFFER_SIZE);
    }
}
