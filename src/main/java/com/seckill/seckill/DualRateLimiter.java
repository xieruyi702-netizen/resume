package com.seckill.seckill;

import java.util.concurrent.ThreadLocalRandom;

/**
 * 双算法 + 分片限流门面。
 * <p>
 * 总配额均分到 N 个分片，请求按线程随机落到一个分片，降低 synchronized 竞争。
 * 每个分片同时持有令牌桶与漏桶，{@link #setMode} 切换生效算法。
 */
public class DualRateLimiter implements RateLimiter {

    public enum Mode { TOKEN, LEAKY }

    /** 分片填充，降低相邻桶伪共享 */
    static final class Shard {
        final TokenBucket token;
        final LeakyBucket leaky;
        @SuppressWarnings("unused")
        private final long p0, p1, p2, p3, p4, p5, p6, p7;

        Shard(long capacity, double ratePerSec) {
            this.token = new TokenBucket(capacity, ratePerSec);
            this.leaky = new LeakyBucket(capacity, ratePerSec);
            this.p0 = this.p1 = this.p2 = this.p3 = this.p4 = this.p5 = this.p6 = this.p7 = 0;
        }
    }

    private volatile Mode mode;
    private volatile Shard[] shards;
    private volatile long totalCapacity;
    private volatile double totalRate;

    public DualRateLimiter(long capacity, double ratePerSec, Mode mode) {
        this(capacity, ratePerSec, mode, 1);
    }

    public DualRateLimiter(long capacity, double ratePerSec, Mode mode, int shardCount) {
        this.mode = mode == null ? Mode.TOKEN : mode;
        this.totalCapacity = capacity;
        this.totalRate = ratePerSec;
        this.shards = buildShards(Math.max(1, shardCount), capacity, ratePerSec);
    }

    private static Shard[] buildShards(int n, long capacity, double ratePerSec) {
        long capEach = Math.max(1, capacity / n);
        long capRem = Math.max(0, capacity - capEach * n);
        double rateEach = ratePerSec / n;
        Shard[] arr = new Shard[n];
        for (int i = 0; i < n; i++) {
            long c = capEach + (i < capRem ? 1 : 0);
            arr[i] = new Shard(c, rateEach);
        }
        return arr;
    }

    public Mode mode() { return mode; }

    public int shardCount() { return shards.length; }

    public long totalCapacity() { return totalCapacity; }

    public double totalRate() { return totalRate; }

    public synchronized void setMode(Mode mode) {
        if (mode != null) this.mode = mode;
    }

    /** 运行时调整分片数（重建桶，统计清零） */
    public synchronized void setShardCount(int shardCount) {
        int n = Math.max(1, Math.min(shardCount, 256));
        this.shards = buildShards(n, totalCapacity, totalRate);
    }

    private Shard pick() {
        Shard[] s = shards;
        if (s.length == 1) return s[0];
        return s[ThreadLocalRandom.current().nextInt(s.length)];
    }

    @Override
    public boolean tryAcquire() {
        Shard s = pick();
        return mode == Mode.LEAKY ? s.leaky.tryAcquire() : s.token.tryAcquire();
    }

    @Override
    public long passed() {
        long sum = 0;
        Mode m = mode;
        for (Shard s : shards) sum += m == Mode.LEAKY ? s.leaky.passed() : s.token.passed();
        return sum;
    }

    @Override
    public long rejected() {
        long sum = 0;
        Mode m = mode;
        for (Shard s : shards) sum += m == Mode.LEAKY ? s.leaky.rejected() : s.token.rejected();
        return sum;
    }

    @Override
    public void resetStats() {
        for (Shard s : shards) {
            s.token.resetStats();
            s.leaky.resetStats();
        }
    }

    @Override
    public synchronized void reconfigure(long capacity, double ratePerSec) {
        this.totalCapacity = capacity;
        this.totalRate = ratePerSec;
        this.shards = buildShards(shards.length, capacity, ratePerSec);
    }

    @Override
    public String name() {
        return (mode == Mode.LEAKY ? "leaky" : "token") + "/shards=" + shards.length;
    }
}
