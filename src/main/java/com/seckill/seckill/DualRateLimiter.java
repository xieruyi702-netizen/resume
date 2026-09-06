package com.seckill.seckill;

/**
 * 双算法限流门面：同时持有令牌桶与漏桶，按 mode 切换生效算法。
 * 压测可对比同容量/同速率下突发通过数与平滑度差异。
 */
public class DualRateLimiter implements RateLimiter {

    public enum Mode { TOKEN, LEAKY }

    private final TokenBucket tokenBucket;
    private final LeakyBucket leakyBucket;
    private volatile Mode mode;

    public DualRateLimiter(long capacity, double ratePerSec, Mode mode) {
        this.tokenBucket = new TokenBucket(capacity, ratePerSec);
        this.leakyBucket = new LeakyBucket(capacity, ratePerSec);
        this.mode = mode == null ? Mode.TOKEN : mode;
    }

    private RateLimiter active() {
        return mode == Mode.LEAKY ? leakyBucket : tokenBucket;
    }

    public Mode mode() { return mode; }

    public synchronized void setMode(Mode mode) {
        if (mode != null) this.mode = mode;
    }

    @Override
    public boolean tryAcquire() { return active().tryAcquire(); }

    @Override
    public long passed() { return active().passed(); }

    @Override
    public long rejected() { return active().rejected(); }

    @Override
    public void resetStats() {
        tokenBucket.resetStats();
        leakyBucket.resetStats();
    }

    @Override
    public void reconfigure(long capacity, double ratePerSec) {
        tokenBucket.reconfigure(capacity, ratePerSec);
        leakyBucket.reconfigure(capacity, ratePerSec);
    }

    @Override
    public String name() { return active().name(); }
}
