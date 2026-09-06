package com.seckill.seckill;

import java.util.concurrent.atomic.AtomicLong;

/**
 * 漏桶限流：容量 capacity，匀速漏出 leakPerSec。
 * 请求到来时向桶内加水；水位超过容量则拒绝。惰性漏水，无需后台线程。
 * 与令牌桶对比：漏桶更强调输出平滑（突发会被水位顶住），令牌桶允许容量内突发。
 */
public class LeakyBucket implements RateLimiter {

    private long capacity;
    private double leakPerNano;
    private double water;
    private long lastLeakNanos;
    private final AtomicLong passed = new AtomicLong();
    private final AtomicLong rejected = new AtomicLong();

    public LeakyBucket(long capacity, double leakPerSec) {
        this.capacity = capacity;
        this.leakPerNano = leakPerSec / 1e9;
        this.water = 0;
        this.lastLeakNanos = System.nanoTime();
    }

    @Override
    public synchronized boolean tryAcquire() {
        leak();
        if (water + 1 <= capacity) {
            water += 1;
            passed.incrementAndGet();
            return true;
        }
        rejected.incrementAndGet();
        return false;
    }

    private void leak() {
        long now = System.nanoTime();
        double drained = (now - lastLeakNanos) * leakPerNano;
        if (drained > 0) {
            water = Math.max(0, water - drained);
            lastLeakNanos = now;
        }
    }

    @Override
    public long passed() { return passed.get(); }

    @Override
    public long rejected() { return rejected.get(); }

    @Override
    public void resetStats() { passed.set(0); rejected.set(0); }

    @Override
    public synchronized void reconfigure(long newCapacity, double leakPerSec) {
        this.water = Math.min(this.water, newCapacity);
        this.capacity = newCapacity;
        this.leakPerNano = leakPerSec / 1e9;
    }

    @Override
    public String name() { return "leaky"; }
}
