package com.seckill.bench;

import com.seckill.seckill.TokenBucket;

import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.LongAdder;

/**
 * 验证伪共享：桶之间插入 long[16] 填充后再测分片扩展。
 * java -cp target/classes com.seckill.bench.RateLimitTune3
 */
public final class RateLimitTune3 {

    static final long TOTAL_CAP = 200;
    static final double TOTAL_RATE = 2000;
    static final long OPS = 8_000_000L;
    static final int REPEAT = 5;

    /** 每个桶独占约 128B+，降低相邻桶伪共享 */
    static final class Padded {
        final TokenBucket bucket;
        final long[] pad = new long[16];
        Padded(TokenBucket b) { this.bucket = b; }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== 无填充 vs 填充（shards=threads，中位数）===");
        for (int n : new int[]{1, 2, 4, 8, 16, 32}) {
            long bare = median(n, false);
            long pad = median(n, true);
            System.out.printf(Locale.ROOT,
                    "  n=%2d  bare=%,d  padded=%,d  (%.2fx)%n",
                    n, bare, pad, pad / (double) bare);
        }
    }

    static long median(int n, boolean padded) throws InterruptedException {
        long[] xs = new long[REPEAT];
        for (int i = 0; i < REPEAT; i++) xs[i] = run(n, padded);
        Arrays.sort(xs);
        return xs[REPEAT / 2];
    }

    static long run(int n, boolean padded) throws InterruptedException {
        long cap = Math.max(1, TOTAL_CAP / n);
        double rate = TOTAL_RATE / n;
        TokenBucket[] buckets = new TokenBucket[n];
        Padded[] holders = padded ? new Padded[n] : null;
        for (int i = 0; i < n; i++) {
            TokenBucket b = new TokenBucket(cap, rate);
            buckets[i] = b;
            if (padded) holders[i] = new Padded(b);
        }
        // 引用 holders 防优化掉
        if (holders != null) {
            long sink = 0;
            for (Padded h : holders) sink += h.pad[0];
            if (sink == 42) System.out.print("");
        }

        long per = OPS / n;
        CountDownLatch gate = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(n);
        LongAdder finished = new LongAdder();

        for (int t = 0; t < n; t++) {
            final int tid = t;
            new Thread(() -> {
                try {
                    gate.await();
                    TokenBucket b = buckets[tid];
                    for (long i = 0; i < per; i++) b.tryAcquire();
                    finished.add(per);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    done.countDown();
                }
            }).start();
        }

        long t0 = System.nanoTime();
        gate.countDown();
        done.await();
        return Math.round(finished.sum() / ((System.nanoTime() - t0) / 1e9));
    }

    private RateLimitTune3() {}
}
