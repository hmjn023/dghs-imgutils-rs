//! Session reuse and process-wide serialization of expensive model compilation.

use super::{InferenceError, SessionKey};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

type SessionEntries<T> = HashMap<SessionKey, Arc<Mutex<T>>>;

/// Keeps cache reads independent of compilation while allowing one initializer.
/// The value is generic so concurrency can be tested without loading a runtime.
pub(super) struct SessionCache<T> {
    entries: Mutex<SessionEntries<T>>,
    initialization: Mutex<()>,
}

impl<T> SessionCache<T> {
    pub(super) fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            initialization: Mutex::new(()),
        }
    }

    /// Serializes all construction, including explicit uncached session APIs.
    pub(super) fn initialize<R>(
        &self,
        initializer: impl FnOnce() -> Result<R, InferenceError>,
    ) -> Result<R, InferenceError> {
        let _initialization = self.initialization.lock().map_err(|error| {
            InferenceError::Initialization(format!("Session initialization lock poisoned: {error}"))
        })?;
        initializer()
    }

    /// Rechecks after waiting so concurrent cache misses never compile twice.
    /// Ordinary initialization errors are not cached and may be retried.
    pub(super) fn get_or_try_init(
        &self,
        key: SessionKey,
        initializer: impl FnOnce() -> Result<T, InferenceError>,
    ) -> Result<Arc<Mutex<T>>, InferenceError> {
        if let Some(session) = self.entries()?.get(&key) {
            return Ok(Arc::clone(session));
        }

        self.initialize(|| {
            {
                let mut entries = self.entries()?;
                if let Some(session) = entries.get(&key) {
                    return Ok(Arc::clone(session));
                }
                // A replaced model must not retain its obsolete compiled graph.
                // Old handles may finish running, but no longer live in the cache.
                entries.retain(|cached, _| {
                    cached.model_path != key.model_path || cached.model_sha256 == key.model_sha256
                });
            }

            // Never hold the entries mutex during provider compilation. Warm
            // models must stay accessible while a different model is loading.
            let session = Arc::new(Mutex::new(initializer()?));
            self.entries()?.insert(key, Arc::clone(&session));
            Ok(session)
        })
    }

    /// Waits for in-flight construction before removing cached entries.
    pub(super) fn clear(&self) -> Result<(), InferenceError> {
        self.initialize(|| {
            self.entries()?.clear();
            Ok(())
        })
    }

    fn entries(&self) -> Result<MutexGuard<'_, SessionEntries<T>>, InferenceError> {
        self.entries.lock().map_err(|error| {
            InferenceError::Initialization(format!("Session cache lock poisoned: {error}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{Backend, Precision};
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    fn key(model: &str) -> SessionKey {
        SessionKey {
            model_path: model.into(),
            model_sha256: "original".into(),
            backend: Backend::Cpu,
            precision: Precision::Auto,
            device_id: 0,
            runtime_fingerprint: "test-runtime".into(),
            provider_fingerprint: "test-provider".into(),
        }
    }

    #[test]
    fn concurrent_requests_share_one_initialization_and_handle() {
        let cache = SessionCache::new();
        let calls = AtomicUsize::new(0);
        let start = Barrier::new(16);
        let sessions = thread::scope(|scope| {
            let handles: Vec<_> = (0..16)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        cache
                            .get_or_try_init(key("model"), || {
                                calls.fetch_add(1, Ordering::SeqCst);
                                thread::yield_now();
                                Ok(42)
                            })
                            .unwrap()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(
            sessions
                .iter()
                .all(|session| Arc::ptr_eq(session, &sessions[0]))
        );
    }

    #[test]
    fn different_models_and_uncached_calls_share_the_compilation_limit() {
        let cache = SessionCache::new();
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let start = Barrier::new(16);
        thread::scope(|scope| {
            for index in 0..16 {
                let cache = &cache;
                let active = &active;
                let peak = &peak;
                let start = &start;
                scope.spawn(move || {
                    start.wait();
                    let initialize = || {
                        let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(current, Ordering::SeqCst);
                        for _ in 0..100 {
                            thread::yield_now();
                        }
                        active.fetch_sub(1, Ordering::SeqCst);
                        Ok(index)
                    };
                    if index % 2 == 0 {
                        cache
                            .get_or_try_init(key(&format!("model-{index}")), initialize)
                            .unwrap();
                    } else {
                        cache.initialize(initialize).unwrap();
                    }
                });
            }
        });
        assert_eq!(peak.load(Ordering::SeqCst), 1);
        assert_eq!(active.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn initialization_failure_can_be_retried() {
        let cache = SessionCache::new();
        assert!(
            cache
                .get_or_try_init(key("model"), || {
                    Err::<usize, _>(InferenceError::Initialization("temporary failure".into()))
                })
                .is_err()
        );
        let session = cache.get_or_try_init(key("model"), || Ok(42)).unwrap();
        assert_eq!(*session.lock().unwrap(), 42);
    }

    #[test]
    fn warm_session_is_accessible_during_another_models_initialization() {
        let cache = SessionCache::new();
        let warm = cache.get_or_try_init(key("warm"), || Ok(1)).unwrap();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (warm_tx, warm_rx) = mpsc::channel();
        thread::scope(|scope| {
            let cache = &cache;
            scope.spawn(move || {
                cache
                    .get_or_try_init(key("cold"), || {
                        entered_tx.send(()).unwrap();
                        release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                        Ok(2)
                    })
                    .unwrap();
            });
            entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            scope.spawn(|| {
                let session = cache
                    .get_or_try_init(key("warm"), || panic!("already cached"))
                    .unwrap();
                warm_tx.send(session).unwrap();
            });
            let result = warm_rx.recv_timeout(Duration::from_secs(2));
            release_tx.send(()).unwrap();
            assert!(Arc::ptr_eq(&result.unwrap(), &warm));
        });
    }

    #[test]
    fn replaced_model_evicts_old_identity_but_preserves_existing_handles() {
        let cache = SessionCache::new();
        let old = cache.get_or_try_init(key("model"), || Ok(1)).unwrap();
        let mut updated_key = key("model");
        updated_key.model_sha256 = "replacement".into();
        let new = cache.get_or_try_init(updated_key, || Ok(2)).unwrap();
        assert!(!Arc::ptr_eq(&old, &new));
        assert_eq!(*old.lock().unwrap(), 1);
        assert_eq!(cache.entries().unwrap().len(), 1);
    }

    #[test]
    fn clear_removes_sessions_without_invalidating_existing_handles() {
        let cache = SessionCache::new();
        let old = cache.get_or_try_init(key("model"), || Ok(1)).unwrap();
        cache.clear().unwrap();
        let new = cache.get_or_try_init(key("model"), || Ok(2)).unwrap();
        assert!(!Arc::ptr_eq(&old, &new));
        assert_eq!(*old.lock().unwrap(), 1);
        assert_eq!(*new.lock().unwrap(), 2);
    }

    #[test]
    fn clear_waits_for_in_flight_initialization_and_does_not_leave_a_session_behind() {
        let cache = SessionCache::new();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (clearing_tx, clearing_rx) = mpsc::channel();
        let (cleared_tx, cleared_rx) = mpsc::channel();
        thread::scope(|scope| {
            let cache = &cache;
            scope.spawn(move || {
                cache
                    .get_or_try_init(key("model"), || {
                        entered_tx.send(()).unwrap();
                        release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                        Ok(1)
                    })
                    .unwrap();
            });
            entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            scope.spawn(move || {
                clearing_tx.send(()).unwrap();
                cache.clear().unwrap();
                cleared_tx.send(()).unwrap();
            });
            clearing_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let early_clear = cleared_rx.recv_timeout(Duration::from_millis(50));
            release_tx.send(()).unwrap();
            assert!(matches!(early_clear, Err(mpsc::RecvTimeoutError::Timeout)));
            cleared_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        });
        assert!(cache.entries().unwrap().is_empty());
    }
}
