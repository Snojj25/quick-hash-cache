use async_chashmap::lru::{Evict, LruCache};

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
