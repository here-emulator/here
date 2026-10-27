use rustc_hash::FxHashMap;
use std::cell::Cell;

use here_types::WordType;

pub trait Cacheable: Clone + Copy {
    /// Log2 of the cached object's minimum size in bytes (e.g., `1` for 2-byte RISC-V instructions),
    ///  not of the cached result (e.g., [`DecodeInstr`]).
    const ADDR_SHIFT_BITS: usize;

    /// Convert cacheable object's address to it's index,
    /// which is address right shifted by [`Self::ADDR_SHIFT_BITS`].
    #[inline]
    fn index_of(addr: WordType) -> usize {
        (addr as usize) >> Self::ADDR_SHIFT_BITS
    }
}

/// A thin warpper that provide hit rate statistics.
pub struct Cache<P: CachePolicy> {
    policy: P,
    hit_count: Cell<u64>,
    access_count: Cell<u64>,
}

impl<P: CachePolicy> Cache<P> {
    #[inline]
    pub fn hit_rate(&self) -> f64 {
        self.hit_count() as f64 / self.access_count() as f64
    }

    #[inline]
    pub fn hit_count(&self) -> u64 {
        self.hit_count.get()
    }

    #[inline]
    pub fn access_count(&self) -> u64 {
        self.access_count.get()
    }
}

impl<P: CachePolicy> CachePolicy for Cache<P> {
    type T = P::T;

    fn new() -> Self {
        Self {
            policy: P::new(),
            hit_count: Cell::new(0),
            access_count: Cell::new(0),
        }
    }

    #[inline]
    fn get(&mut self, addr: WordType) -> Option<Self::T> {
        self.access_count.update(|cnt| cnt + 1);
        self.policy
            .get(addr)
            .inspect(|_| self.hit_count.update(|cnt| cnt + 1))
    }

    #[inline]
    fn put(&mut self, addr: WordType, data: Self::T) {
        self.policy.put(addr, data)
    }

    #[inline]
    fn invalidate(&mut self, addr: WordType) {
        self.policy.invalidate(addr);
    }

    #[inline]
    fn clear(&mut self) {
        self.policy.clear();
    }
}

pub trait CachePolicy {
    type T: Cacheable;

    fn new() -> Self;
    fn get(&mut self, addr: WordType) -> Option<Self::T>;
    fn put(&mut self, addr: WordType, data: Self::T);
    fn invalidate(&mut self, addr: WordType);
    fn clear(&mut self);
}

pub struct DirectCache<T, const N: usize> {
    cache: Box<[(WordType, Option<T>); N]>,
}

impl<T: Cacheable, const N: usize> DirectCache<T, N> {
    #[inline]
    fn get_id(addr: WordType) -> usize {
        T::index_of(addr) & (N - 1)
    }
}

impl<T: Cacheable, const N: usize> CachePolicy for DirectCache<T, N> {
    type T = T;

    fn new() -> Self {
        debug_assert!(N > 0 && (N & (N - 1)) == 0, "N must be a power of two");
        Self {
            cache: unsafe { Box::new_zeroed().assume_init() },
        }
    }

    #[inline]
    fn get(&mut self, addr: WordType) -> Option<T> {
        let (tag, data) = &self.cache[Self::get_id(addr)];
        if *tag == addr { data.clone() } else { None }
    }

    #[inline]
    fn put(&mut self, addr: WordType, data: T) {
        self.cache[Self::get_id(addr)] = (addr, Some(data));
    }

    #[inline]
    fn invalidate(&mut self, addr: WordType) {
        self.cache[Self::get_id(addr)] = (0, None);
    }

    #[inline]
    fn clear(&mut self) {
        self.cache.fill((0, None));
    }
}

/// Helper struct for set-associative cache, representing one set with W ways.
#[derive(Clone)]
struct CacheSet<T: Cacheable, const W: usize> {
    idx: usize,
    slots: [(WordType, Option<T>); W],
}

impl<T: Cacheable, const W: usize> CacheSet<T, W> {
    fn new() -> Self {
        Self {
            idx: 0,
            slots: [(0, None); W],
        }
    }

    #[inline]
    fn insert(&mut self, addr: WordType, data: T) {
        let (src, value) = &mut self.slots[self.idx];
        *src = addr;
        *value = Some(data);

        self.idx += 1;
        if self.idx == W {
            self.idx = 0;
        }
    }

    #[inline]
    fn invalidate(&mut self, target: WordType) {
        if let Some(index) = self.slots.iter().position(|&(addr, _v)| addr == target) {
            self.slots[index] = (0, None);
        }
    }
}

/// Set-associative cache with S sets and W ways per set.
///
/// TODO: Current replacement policy is FIFO, which is not optimal.
///
/// Example:
/// ```text
/// SetCache<DecodeInstr, 4, 2> // 4 sets, 2 ways per set
/// ```
pub struct SetCache<T: Cacheable, const S: usize, const W: usize> {
    cache: Box<[CacheSet<T, W>]>,
}

impl<T: Cacheable, const S: usize, const W: usize> SetCache<T, S, W> {
    #[inline]
    fn set_index_of(addr: WordType) -> usize {
        (T::index_of(addr)) & (S - 1)
    }
}

impl<T: Cacheable, const S: usize, const W: usize> CachePolicy for SetCache<T, S, W> {
    type T = T;

    fn new() -> Self {
        debug_assert!(S > 0 && (S & (S - 1)) == 0, "S must be a power of two.");
        debug_assert!(W > 0 && (W & (W - 1)) == 0, "W must be a power of two.");

        Self {
            cache: vec![CacheSet::new(); S].into_boxed_slice(),
        }
    }

    #[inline]
    fn get(&mut self, addr: WordType) -> Option<T> {
        let set = &self.cache[Self::set_index_of(addr)];

        set.slots
            .iter()
            .position(|&(a, _val)| a == addr)
            .and_then(|index| set.slots[index].1.clone())
    }

    #[inline]
    fn put(&mut self, addr: WordType, data: T) {
        self.cache[Self::set_index_of(addr)].insert(addr, data);
    }

    #[inline]
    fn invalidate(&mut self, addr: WordType) {
        self.cache[Self::set_index_of(addr)].invalidate(addr);
    }

    #[inline]
    fn clear(&mut self) {
        *self = Self::new()
    }
}

/// Used to test the performance of other cache implementations.
pub struct NullCache<T: Cacheable> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Cacheable> CachePolicy for NullCache<T> {
    type T = T;

    fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    fn get(&mut self, _addr: WordType) -> Option<T> {
        None
    }

    fn put(&mut self, _addr: WordType, _data: T) {}

    fn invalidate(&mut self, _addr: WordType) {}

    fn clear(&mut self) {}
}

pub type FxCache<T> = FxHashMap<WordType, T>;

impl<T: Cacheable> CachePolicy for FxHashMap<WordType, T> {
    type T = T;

    fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn get(&mut self, addr: WordType) -> Option<T> {
        FxHashMap::get(self, &addr).copied()
    }

    #[inline]
    fn put(&mut self, addr: WordType, data: T) {
        self.insert(addr, data);
    }

    #[inline]
    fn invalidate(&mut self, addr: WordType) {
        self.remove(&addr);
    }

    #[inline]
    fn clear(&mut self) {
        self.clear();
    }
}

/// Chained two-level cache policy.
pub struct ChainedCache<L1: CachePolicy, L2: CachePolicy<T = L1::T>> {
    l1: L1,
    l2: L2,
}

impl<L1: CachePolicy, L2: CachePolicy<T = L1::T>> CachePolicy for ChainedCache<L1, L2> {
    type T = L1::T;

    fn new() -> Self {
        Self {
            l1: L1::new(),
            l2: L2::new(),
        }
    }

    #[inline]
    fn get(&mut self, addr: WordType) -> Option<Self::T> {
        if let Some(data) = self.l1.get(addr) {
            return Some(data);
        }
        let data = self.l2.get(addr)?;
        self.l1.put(addr, data);
        Some(data)
    }

    #[inline]
    fn put(&mut self, addr: WordType, data: Self::T) {
        self.l1.put(addr, data);
        self.l2.put(addr, data);
    }

    #[inline]
    fn invalidate(&mut self, addr: WordType) {
        self.l1.invalidate(addr);
        self.l2.invalidate(addr);
    }

    #[inline]
    fn clear(&mut self) {
        self.l1.clear();
        self.l2.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct MockCacheable(u32);

    impl Cacheable for MockCacheable {
        const ADDR_SHIFT_BITS: usize = 0;
    }

    fn test_cache_common<C: CachePolicy<T = MockCacheable>>() {
        let mut cache = C::new();

        assert_eq!(cache.get(0), None);

        cache.put(0, MockCacheable(42));
        assert_eq!(cache.get(0), Some(MockCacheable(42)));

        cache.invalidate(0);
        assert_eq!(cache.get(0), None);

        cache.put(1, MockCacheable(100));
        cache.put(2, MockCacheable(200));
        assert_eq!(cache.get(1), Some(MockCacheable(100)));
        assert_eq!(cache.get(2), Some(MockCacheable(200)));

        cache.invalidate(1);
        assert_eq!(cache.get(1), None);

        cache.clear();
        assert_eq!(cache.get(1), None);
        assert_eq!(cache.get(2), None);
    }

    #[test]
    fn common_cache_tests() {
        test_cache_common::<DirectCache<MockCacheable, 8>>();
        test_cache_common::<SetCache<MockCacheable, 4, 2>>();
        test_cache_common::<FxCache<MockCacheable>>();
        test_cache_common::<ChainedCache<DirectCache<MockCacheable, 8>, FxCache<MockCacheable>>>();
    }

    #[test]
    fn set_cache_test() {
        let mut cache = SetCache::<MockCacheable, 4, 2>::new();

        for i in 0..4 {
            cache.put(i, MockCacheable(i as u32));
        }

        cache.put(8, MockCacheable(8));

        for i in 0..4 {
            assert_eq!(cache.get(i), Some(MockCacheable(i as u32)));
        }

        assert_eq!(cache.get(8), Some(MockCacheable(8)));
    }

    #[test]
    fn test_chained_cache_keeps_entries_evicted_from_l1() {
        let mut cache =
            ChainedCache::<DirectCache<MockCacheable, 2>, FxCache<MockCacheable>>::new();

        cache.put(0, MockCacheable(10));
        cache.put(1, MockCacheable(20));
        assert_eq!(cache.get(0), Some(MockCacheable(10)));
        assert_eq!(cache.get(1), Some(MockCacheable(20)));

        // Address 2 collides with address 0 in the 2-slot L1, evicting it there.
        cache.put(2, MockCacheable(30));

        // The entry must still be retrievable from L2.
        assert_eq!(cache.get(0), Some(MockCacheable(10)));
        assert_eq!(cache.get(2), Some(MockCacheable(30)));
    }
}
