use std::{borrow::Borrow, fmt};

use hashbrown::raw::RawTable;

#[derive(Debug, Clone, Copy)]
pub struct Bucket<K, V> {
    pub(crate) hash: u64,
    pub(crate) key: K,
    pub(crate) value: V,
}

pub struct IndexedShard<K, V> {
    pub(crate) indices: RawTable<usize>,
    pub(crate) entries: Vec<Bucket<K, V>>,
}

impl<K, V> Clone for IndexedShard<K, V>
where
    K: Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        let indices = self.indices.clone();
        let mut entries = Vec::with_capacity(indices.len());
        entries.clone_from(&self.entries);
        IndexedShard { indices, entries }
    }

    fn clone_from(&mut self, source: &Self) {
        self.indices
            .clone_from_with_hasher(&source.indices, |&idx| source.entries[idx].hash);

        if self.entries.capacity() < source.entries.len() {
            // If we must resize, match the indices capacity
            self.reserve_entries();
        }

        self.entries.clone_from(&source.entries);
    }
}

impl<K, V> fmt::Debug for IndexedShard<K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct DebugIndices<'a>(pub &'a RawTable<usize>);
        impl fmt::Debug for DebugIndices<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                // SAFETY: we're not letting any of the buckets escape this function
                let indices = unsafe { self.0.iter().map(|raw_bucket| *raw_bucket.as_ref()) };
                f.debug_list().entries(indices).finish()
            }
        }

        f.debug_struct("IndexMapCore")
            .field("indices", &DebugIndices(&self.indices))
            .field("entries", &self.entries)
            .finish()
    }
}

impl<K, V> IndexedShard<K, V> {
    #[inline]
    pub const fn new() -> Self {
        IndexedShard {
            indices: RawTable::new(),
            entries: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.indices.len()
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.indices.clear();
    }

    /// Append a key-value pair, *without* checking whether it already exists,
    /// and return the pair's new index.
    #[inline]
    fn push(&mut self, hash: u64, key: K, value: V) -> usize {
        let index = self.entries.len();

        let IndexedShard {
            ref mut indices,
            ref entries,
        } = self;

        indices.insert(hash, index, |&idx| unsafe { entries.get_unchecked(idx).hash });

        if index == self.entries.capacity() {
            // Reserve our own capacity synced to the indices,
            // rather than letting `Vec::push` just double it.
            self.reserve_entries();
        }

        self.entries.push(Bucket { hash, key, value });

        index
    }

    /// Return the index in `entries` where an equivalent key can be found
    #[inline]
    pub(crate) fn get_index_of<Q: ?Sized>(&self, hash: u64, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Eq,
    {
        self.indices
            .get(hash, |&idx| self.entries[idx].key.borrow() == key)
            .copied()
    }

    #[inline]
    pub(crate) fn get<Q: ?Sized>(&self, hash: u64, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq,
    {
        self.get_index_of(hash, key)
            .map(|idx| unsafe { &self.entries.get_unchecked(idx).value })
    }

    #[inline]
    pub(crate) fn get_mut<Q: ?Sized>(&mut self, hash: u64, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Eq,
    {
        self.get_index_of(hash, key)
            .map(move |idx| unsafe { &mut self.entries.get_unchecked_mut(idx).value })
    }

    #[inline]
    pub(crate) fn insert_full(
        &mut self,
        hash: u64,
        key: K,
        value: V,
        before_insert: impl FnOnce(),
    ) -> (usize, Option<V>)
    where
        K: Eq,
    {
        match self.get_index_of(hash, &key) {
            Some(i) => (i, Some(std::mem::replace(&mut self.entries[i].value, value))),
            None => {
                before_insert();
                (self.push(hash, key, value), None)
            }
        }
    }

    #[inline]
    pub(crate) fn swap_remove_full<Q: ?Sized>(&mut self, hash: u64, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Eq,
    {
        match self.get_index_of(hash, key) {
            Some(index) => {
                self.indices.erase_entry(hash, |&idx| idx == index);
                Some(self.swap_remove_finish(index))
            }
            None => None,
        }
    }

    #[inline]
    pub(crate) unsafe fn swap_remove_index_raw(&mut self, index: usize) -> (K, V) {
        debug_assert!(index < self.len(), "index {} < {} length", index, self.len());

        let hash = self.entries.get_unchecked(index).hash;

        self.indices.erase_entry(hash, |&idx| idx == index);
        self.swap_remove_finish(index)
    }

    #[inline]
    fn swap_remove_finish(&mut self, index: usize) -> (K, V) {
        // use swap_remove, but then we need to update the index that points
        // to the other entry that has to move
        let entry = self.entries.swap_remove(index);

        // correct index that points to the entry that had to swap places
        if let Some(entry) = self.entries.get(index) {
            // was not last element, examine new element in `index` and find it in indices
            let last = self.entries.len();

            *self
                .indices
                .get_mut(entry.hash, |&idx| idx == last)
                .expect("index not found") = index;
        }

        (entry.key, entry.value)
    }

    /// Reserve entries capacity to match the indices
    #[inline]
    fn reserve_entries(&mut self) {
        let additional = self.indices.capacity() - self.entries.len();
        self.entries.reserve_exact(additional);
    }

    pub(crate) fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        let mut i = 0;
        while i < self.len() {
            let Bucket { key, value, hash } = unsafe { self.entries.get_unchecked_mut(i) };

            if f(key, value) {
                i += 1;
            } else {
                self.indices.erase_entry(*hash, |&idx| idx == i);
                self.swap_remove_finish(i);
            }
        }
    }
}
