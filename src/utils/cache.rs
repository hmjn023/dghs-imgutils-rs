use lru::LruCache;
use std::fmt::Debug;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::Mutex;

pub struct ThreadSafeLruCache<K, V> {
    inner: Mutex<LruCache<K, V>>,
}

impl<K: Eq + Hash + Debug, V: Clone> ThreadSafeLruCache<K, V> {
    pub fn new(maxsize: usize) -> Self {
        let cap = NonZeroUsize::new(maxsize.max(1)).unwrap_or(NonZeroUsize::MIN);
        Self {
            inner: Mutex::new(LruCache::new(cap)),
        }
    }

    pub fn get_or_put<F>(&self, key: K, f: F) -> V
    where
        F: FnOnce() -> V,
    {
        let mut cache = self.inner.lock().unwrap();
        if let Some(value) = cache.get(&key) {
            return value.clone();
        }
        let value = f();
        cache.put(key, value.clone());
        value
    }

    pub fn get(&self, key: &K) -> Option<V> {
        self.inner.lock().unwrap().get(key).cloned()
    }

    pub fn put(&self, key: K, value: V) {
        let mut cache = self.inner.lock().unwrap();
        cache.put(key, value);
    }

    pub fn clear(&self) {
        self.inner.lock().unwrap().clear();
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheLevel {
    Global,
    Process,
    Thread,
}

impl CacheLevel {
    pub fn context_key(&self) -> Option<ContextKey> {
        match self {
            CacheLevel::Global => None,
            CacheLevel::Process => Some(ContextKey::Process(std::process::id())),
            CacheLevel::Thread => Some(ContextKey::Thread(
                std::process::id(),
                std::thread::current().id(),
            )),
        }
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum ContextKey {
    Process(u32),
    Thread(u32, std::thread::ThreadId),
}

pub type LevelTyping = CacheLevel;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let cache: ThreadSafeLruCache<String, i32> = ThreadSafeLruCache::new(10);
        assert_eq!(cache.get_or_put("a".to_string(), || 42), 42);
        assert_eq!(cache.get_or_put("a".to_string(), || 99), 42);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_eviction() {
        let cache: ThreadSafeLruCache<i32, i32> = ThreadSafeLruCache::new(2);
        assert_eq!(cache.get_or_put(1, || 10), 10);
        assert_eq!(cache.get_or_put(2, || 20), 20);
        assert_eq!(cache.get_or_put(3, || 30), 30);
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&1).is_none());
    }

    #[test]
    fn test_cache_level() {
        assert_eq!(CacheLevel::Global.context_key(), None);
        assert!(CacheLevel::Process.context_key().is_some());
        assert!(CacheLevel::Thread.context_key().is_some());
    }
}
