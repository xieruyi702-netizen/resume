package com.seckill.seckill;

import java.util.concurrent.atomic.AtomicLong;

/**
 * 令牌桶限流：容量 capacity，匀速补充 refillPerSec。
 * syncaddToken 计算自上次补充至今应补令牌数（惰性补充，无需后台线程）。
 * 允许桶内突发，适合秒杀开闸瞬间的短时高峰。
 */
public class TokenBucket implements RateLimiter {

    private long capacity;
    private double refillPerNano;
    private double tokens;
    private long lastRefillNanos;
    private final AtomicLong passed = new AtomicLong();
    private final AtomicLong rejected = new AtomicLong();

    public TokenBucket(long capacity, double refillPerSec) {
        this.capacity = capacity;
        this.refillPerNano = refillPerSec / 1e9;
        this.tokens = capacity;
        this.lastRefillNanos = System.nanoTime();
    }

    /** 尝试获取 1 个令牌；拿不到立即拒绝（fail-fast，不排队） */
    @Override
    public synchronized boolean tryAcquire() {
        refill();
        if (tokens >= 1) {
            tokens -= 1;
            passed.incrementAndGet();
            return true;
        }
        rejected.incrementAndGet();
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
    public long passed() { return passed.get(); }

    @Override
    public long rejected() { return rejected.get(); }

    @Override
    public void resetStats() { passed.set(0); rejected.set(0); }

    /** 压测/运维用：运行时调整桶参数 */
    @Override
    public synchronized void reconfigure(long newCapacity, double refillPerSec) {
        this.tokens = Math.min(this.tokens, newCapacity);
        this.capacity = newCapacity;
        this.refillPerNano = refillPerSec / 1e9;
    }

    @Override
    public String name() { return "token"; }
}
