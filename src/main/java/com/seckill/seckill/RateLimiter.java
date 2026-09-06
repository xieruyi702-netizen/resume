package com.seckill.seckill;

/**
 * 秒杀入口限流抽象：令牌桶 / 漏桶可切换，便于对比两种算法在突发与平滑吞吐上的差异。
 */
public interface RateLimiter {

    /** 尝试获取 1 个配额；拿不到立即拒绝（fail-fast，不排队） */
    boolean tryAcquire();

    long passed();

    long rejected();

    void resetStats();

    /** 运行时调整容量与速率（令牌桶=补充速率，漏桶=漏出速率） */
    void reconfigure(long capacity, double ratePerSec);

    String name();
}
