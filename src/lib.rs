use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

pub use hashbrown::hash_map::DefaultHashBuilder;
use hashbrown::hash_map::{HashMap, RawEntryMut};

use tokio::sync::{OwnedRwLockMappedWriteGuard, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

pub mod lru;

#[derive(Debug)]
pub struct CHashMap<K, T, S = DefaultHashBuilder> {
    hash_builder: S,
    shards: Vec<Arc<RwLock<HashMap<K, T, S>>>>,
    size: AtomicUsize,
}

impl<K, T> CHashMap<K, T, DefaultHashBuilder> {
    pub fn new(num_shards: usize) -> Self {
        Self::with_hasher(num_shards, DefaultHashBuilder::default())
    }
}

impl<K, T> Default for CHashMap<K, T, DefaultHashBuilder> {
    fn default() -> Self {
        Self::new(num_cpus::get())
    }
}

#[doc(hidden)]
pub trait Erased {}
impl<T> Erased for T {}

pub type ReadHandle<T, U> = OwnedRwLockReadGuard<T, U>;
pub type WriteHandle<T, U> = OwnedRwLockMappedWriteGuard<T, U>;

pub type Shard<K, T, S> = HashMap<K, T, S>;

impl<K, T, S> CHashMap<K, T, S>
where
    S: Clone,
{
    pub fn with_hasher(num_shards: usize, hash_builder: S) -> Self {
        CHashMap {
            shards: (0..num_shards)
                .into_iter()
                .map(|_| Arc::new(RwLock::new(HashMap::with_hasher(hash_builder.clone()))))
                .collect(),
            hash_builder,
            size: AtomicUsize::new(0),
        }
    }
}

impl<K, T, S> CHashMap<K, T, S>
where
    K: Clone,
    T: Clone,
    S: Clone,
{
    /// Duplicates/Clones the CHashMap. A CHashMap cannot be cloned regularly due to internal async locking.
    pub async fn duplicate(&self) -> Self {
        let mut shards = Vec::with_capacity(self.shards.len());
        let mut size = 0;

        for shard in &self.shards {
            let shard = shard.read().await.clone();
            size += shard.len();
            shards.push(Arc::new(RwLock::new(shard)));
        }

        CHashMap {
            shards,
            hash_builder: self.hash_builder.clone(),
            size: AtomicUsize::new(size),
        }
    }
}

impl<K, T, S> CHashMap<K, T, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    pub fn hash_builder(&self) -> &S {
        &self.hash_builder
    }

    #[inline]
    fn hash_and_shard<Q: ?Sized>(&self, key: &Q) -> (u64, usize)
    where
        Q: Hash + Eq,
    {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        (hash, hash as usize % self.shards.len())
    }

    pub async fn clear(&self) {
        for shard in &self.shards {
            let mut shard = shard.write().await;

            let len = shard.len();
            shard.clear();

            self.size.fetch_sub(len, Ordering::SeqCst);
        }
    }

    pub async fn retain<F>(&self, f: F)
    where
        F: Fn(&K, &mut T) -> bool,
    {
        for shard in &self.shards {
            let mut shard = shard.write().await;

            let len = shard.len();
            shard.retain(&f);

            self.size.fetch_sub(len - shard.len(), Ordering::SeqCst);
        }
    }

    pub fn iter_shards<'a>(&'a self) -> impl Iterator<Item = &'a RwLock<Shard<K, T, S>>> {
        self.shards.iter().map(|s| &**s)
    }

    pub fn size(&self) -> usize {
        self.size.load(Ordering::SeqCst)
    }

    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    pub fn try_maybe_contains_hash(&self, hash: u64) -> bool {
        let shard_idx = hash as usize % self.shards.len();
        let shard = unsafe { self.shards.get_unchecked(shard_idx) };

        if let Ok(shard) = shard.try_read() {
            shard.raw_entry().from_hash(hash, |_| true).is_some()
        } else {
            false
        }
    }

    pub async fn contains_hash(&self, hash: u64) -> bool {
        let shard_idx = hash as usize % self.shards.len();
        let shard = unsafe { self.shards.get_unchecked(shard_idx) };

        shard.read().await.raw_entry().from_hash(hash, |_| true).is_some()
    }

    pub async fn contains<Q: ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.contains_hash(self.hash_and_shard(key).0).await
    }

    pub async fn remove<Q: ?Sized>(&self, key: &Q) -> Option<T>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let (hash, shard_idx) = self.hash_and_shard(&key);
        let mut shard = unsafe { self.shards.get_unchecked(shard_idx).write().await };

        match shard.raw_entry_mut().from_key_hashed_nocheck(hash, key) {
            RawEntryMut::Occupied(occupied) => {
                let value = occupied.remove();
                self.size.fetch_sub(1, Ordering::SeqCst);
                Some(value)
            }
            RawEntryMut::Vacant(_) => None,
        }
    }

    pub async fn insert(&self, key: K, value: T) -> Option<T> {
        let (hash, shard_idx) = self.hash_and_shard(&key);
        let mut shard = unsafe { self.shards.get_unchecked(shard_idx).write().await };

        match shard.raw_entry_mut().from_key_hashed_nocheck(hash, &key) {
            RawEntryMut::Occupied(mut occupied) => Some(occupied.insert(value)),
            RawEntryMut::Vacant(vacant) => {
                self.size.fetch_add(1, Ordering::SeqCst);
                vacant.insert_hashed_nocheck(hash, key, value);
                None
            }
        }
    }

    pub async fn get<Q: ?Sized>(&self, key: &Q) -> Option<ReadHandle<impl Erased, T>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let (hash, shard_idx) = self.hash_and_shard(key);
        let shard = unsafe { self.shards.get_unchecked(shard_idx).clone().read_owned().await };

        OwnedRwLockReadGuard::try_map(shard, |shard| {
            match shard.raw_entry().from_key_hashed_nocheck(hash, key) {
                Some((_, value)) => Some(value),
                None => None,
            }
        })
        .ok()
    }

    pub async fn get_cloned<Q: ?Sized>(&self, key: &Q) -> Option<T>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
        T: Clone,
    {
        let (hash, shard_idx) = self.hash_and_shard(key);
        let shard = unsafe { self.shards.get_unchecked(shard_idx).clone().read_owned().await };

        match shard.raw_entry().from_key_hashed_nocheck(hash, key) {
            Some((_, value)) => Some(value.clone()),
            None => None,
        }
    }

    pub async fn get_mut<Q: ?Sized>(&self, key: &Q) -> Option<WriteHandle<impl Erased, T>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let (hash, shard_idx) = self.hash_and_shard(key);
        let shard = unsafe { self.shards.get_unchecked(shard_idx).clone().write_owned().await };

        OwnedRwLockWriteGuard::try_map(shard, |shard| {
            match shard.raw_entry_mut().from_key_hashed_nocheck(hash, key) {
                RawEntryMut::Occupied(occupied) => Some(occupied.into_mut()),
                RawEntryMut::Vacant(_) => None,
            }
        })
        .ok()
    }

    pub async fn get_or_insert(&self, key: &K, on_insert: impl FnOnce() -> T) -> ReadHandle<impl Erased, T>
    where
        K: Clone,
    {
        let (hash, shard_idx) = self.hash_and_shard(key);
        let mut shard = unsafe { self.shards.get_unchecked(shard_idx).clone().write_owned().await };

        if let RawEntryMut::Vacant(vacant) = shard.raw_entry_mut().from_key_hashed_nocheck(hash, key) {
            self.size.fetch_add(1, Ordering::SeqCst);

            vacant.insert_hashed_nocheck(hash, key.clone(), on_insert());
        }

        // TODO: Having to do another lookup for a read-reference is wasteful, maybe use an alternate custom ReadHandle?
        OwnedRwLockReadGuard::map(OwnedRwLockWriteGuard::downgrade(shard), |shard| {
            match shard.raw_entry().from_key_hashed_nocheck(hash, key) {
                Some((_, value)) => value,
                None => unreachable!(),
            }
        })
    }

    pub async fn get_mut_or_insert(
        &self,
        key: &K,
        on_insert: impl FnOnce() -> T,
    ) -> WriteHandle<impl Erased, T>
    where
        K: Clone,
    {
        let (hash, shard_idx) = self.hash_and_shard(key);
        let shard = unsafe { self.shards.get_unchecked(shard_idx).clone().write_owned().await };

        OwnedRwLockWriteGuard::map(shard, |shard| {
            shard
                .raw_entry_mut()
                .from_key_hashed_nocheck(hash, key)
                .or_insert_with(|| {
                    self.size.fetch_add(1, Ordering::SeqCst);

                    (key.clone(), on_insert())
                })
                .1
        })
    }

    pub async fn get_or_default(&self, key: &K) -> ReadHandle<impl Erased, T>
    where
        K: Clone,
        T: Default,
    {
        self.get_or_insert(key, Default::default).await
    }

    pub async fn get_mut_or_default(&self, key: &K) -> WriteHandle<impl Erased, T>
    where
        K: Clone,
        T: Default,
    {
        self.get_mut_or_insert(key, Default::default).await
    }

    /*
    pub async fn shard_mut<Q: ?Sized>(&self, key: &Q) -> WriteLock<K, T, S, Shard<K, T, S>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let (_, shard_idx) = self.hash_and_shard(key);
        let shard = unsafe { self.shards.get_unchecked(shard_idx).clone().write_owned().await };

        OwnedRwLockWriteGuard::map(shard, |shard| shard)
    }

    pub async fn entry<Q: ?Sized>(&self, key: &Q) -> WriteHandle<impl Erased, Entry<'_, K, T, S>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let (hash, shard_idx) = self.hash_and_shard(key);
        let shard = unsafe { self.shards.get_unchecked(shard_idx).clone().write_owned().await };

        OwnedRwLockWriteGuard::map(shard, |shard| {
            shard.raw_entry_mut().from_key_hashed_nocheck(hash, key)
        })
    }
    */

    /// Aggregates all the provided keys and batches together access to the underlying shards,
    /// reducing locking overhead at the cost of memory to buffer keys/hashes
    pub async fn batch_read<'a, Q: 'a + ?Sized, I, F>(
        &self,
        keys: I,
        cache: Option<&mut Vec<(&'a Q, u64, usize)>>,
        mut f: F,
    ) where
        K: Borrow<Q>,
        Q: Hash + Eq,
        I: IntoIterator<Item = &'a Q>,
        F: FnMut(&'a Q, Option<(&K, &T)>),
    {
        let mut own_cache = Vec::new();
        let cache = match cache {
            Some(cache) => {
                cache.clear();
                cache
            }
            None => &mut own_cache,
        };

        cache.extend(keys.into_iter().map(|key| {
            let (hash, shard) = self.hash_and_shard(key);
            (key, hash, shard)
        }));

        if cache.is_empty() {
            return;
        }

        cache.sort_unstable_by_key(|(_, _, shard)| *shard);

        let mut i = 0;
        'outer: loop {
            let current_shard = cache[i].2;
            let shard = unsafe { self.shards.get_unchecked(current_shard).read().await };

            while cache[i].2 == current_shard {
                f(
                    cache[i].0,
                    shard.raw_entry().from_key_hashed_nocheck(cache[i].1, cache[i].0),
                );
                i += 1;

                if i >= cache.len() {
                    break 'outer;
                }
            }
        }

        cache.clear();
    }

    /// Aggregates all the provided keys and batches together access to the underlying shards,
    /// reducing locking overhead at the cost of memory to buffer keys/hashes
    pub async fn batch_write<'a, Q: 'a + ?Sized, I, F>(
        &self,
        keys: I,
        cache: Option<&mut Vec<(&'a Q, u64, usize)>>,
        mut f: F,
    ) where
        K: Borrow<Q>,
        Q: Hash + Eq,
        I: IntoIterator<Item = &'a Q>,
        F: FnMut(&'a Q, hashbrown::hash_map::RawEntryMut<K, T, S>),
    {
        let mut own_cache = Vec::new();
        let cache = match cache {
            Some(cache) => {
                cache.clear();
                cache
            }
            None => &mut own_cache,
        };

        cache.extend(keys.into_iter().map(|key| {
            let (hash, shard) = self.hash_and_shard(key);
            (key, hash, shard)
        }));

        if cache.is_empty() {
            return;
        }

        cache.sort_unstable_by_key(|(_, _, shard)| *shard);

        let mut i = 0;
        'outer: loop {
            let current_shard = cache[i].2;
            let mut shard = unsafe { self.shards.get_unchecked(current_shard).write().await };

            while cache[i].2 == current_shard {
                f(
                    cache[i].0,
                    shard
                        .raw_entry_mut()
                        .from_key_hashed_nocheck(cache[i].1, cache[i].0),
                );
                i += 1;

                if i >= cache.len() {
                    break 'outer;
                }
            }
        }

        cache.clear();
    }
}
