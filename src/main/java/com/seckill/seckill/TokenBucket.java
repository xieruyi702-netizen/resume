package com.seckill.seckill;

import java.util.concurrent.atomic.LongAdder;

/**
 * 令牌桶限流：容量 capacity，匀速补充 refillPerSec。
 * 惰性补充，无需后台线程；允许桶内突发。
 */
public class TokenBucket implements RateLimiter {

    private long capacity;
    private double refillPerNano;
    private double tokens;
    private long lastRefillNanos;
    private final LongAdder passed = new LongAdder();
    private final LongAdder rejected = new LongAdder();

    public TokenBucket(long capacity, double refillPerSec) {
        this.capacity = capacity;
        this.refillPerNano = refillPerSec / 1e9;
        this.tokens = capacity;
        this.lastRefillNanos = System.nanoTime();
    }

    @Override
    public synchronized boolean tryAcquire() {
        refill();
        if (tokens >= 1) {
            tokens -= 1;
            passed.increment();
            return true;
        }
        rejected.increment();
        return false;
    }

    private void refill() {
        long now = System.nanoTime();
        double add = (now - lastRefillNanos) * refillPerNano;
        if (add > 0) {
            tokens = Math.min(capacity, tokens + add);
            lastRefillNanos = now;
        }
    }

    @Override
    public long passed() { return passed.sum(); }

    @Override
    public long rejected() { return rejected.sum(); }

    @Override
    public void resetStats() { passed.reset(); rejected.reset(); }

    @Override
    public synchronized void reconfigure(long newCapacity, double refillPerSec) {
        this.tokens = Math.min(this.tokens, newCapacity);
        this.capacity = newCapacity;
        this.refillPerNano = refillPerSec / 1e9;
    }

    @Override
    public String name() { return "token"; }
}
