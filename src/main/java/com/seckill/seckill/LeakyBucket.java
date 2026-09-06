package com.seckill.seckill;

import java.util.concurrent.atomic.LongAdder;

/**
 * 漏桶限流：容量 capacity，匀速漏出 leakPerSec。
 * 惰性漏水；突发会被水位顶住，出水更平滑。
 */
public class LeakyBucket implements RateLimiter {

    private long capacity;
    private double leakPerNano;
    private double water;
    private long lastLeakNanos;
    private final LongAdder passed = new LongAdder();
    private final LongAdder rejected = new LongAdder();

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
            passed.increment();
            return true;
        }
        rejected.increment();
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
    public long passed() { return passed.sum(); }

    @Override
    public long rejected() { return rejected.sum(); }

    @Override
    public void resetStats() { passed.reset(); rejected.reset(); }

    @Override
    public synchronized void reconfigure(long newCapacity, double leakPerSec) {
        this.water = Math.min(this.water, newCapacity);
        this.capacity = newCapacity;
        this.leakPerNano = leakPerSec / 1e9;
    }

    @Override
    public String name() { return "leaky"; }
}
