# quick-hash-cache

An Async Concurrent Hashmap with LRU
Build a blazing fast HashMap Cache structure that can be used for web servers, databases, or other systems with high levels of concurrent access.
It improves on the mostly single threaded nature of Redis and supports advanced custom data structures.

### Example

```rust
use quick_hash_cache::lru::{Evict, LruCache};

#[tokio::main]
async fn main() {
    let cache = LruCache::default();

    for i in 0..20 {
        cache.insert(i, i).await;
    }

    println!("{:?}", cache.get(&2).await);

    let res = cache.evict_many(10, rand::thread_rng()).await;

    assert_eq!(res.len(), 10);

    println!("{:?}", res);
    println!("{} {:?}", cache.size(), cache.get(&0).await);
    println!("{} {:?}", cache.size(), cache.get(&2).await);
}
```

### Methods

hash_builder,
hash_and_shard,
clear,
retain,
iter_shards,
size,
num_shards,
try_maybe_contains_hash,
contains_hash,
contains,
remove,
insert,
get,
get_cloned,
get_mut,
get_or_insert,
get_mut_or_insert,
get_or_default,
get_mut_or_default,
batch_read,
batch_write
