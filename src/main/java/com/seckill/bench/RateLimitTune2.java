package com.seckill.bench;

import com.seckill.seckill.TokenBucket;

import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.LongAdder;

/**
 * 聚焦：一分片绑一线程（无锁竞争）vs 单桶多线程。
 * 每配置跑 5 次取中位数。
 *
 * java -cp target/classes com.seckill.bench.RateLimitTune2
 */
public final class RateLimitTune2 {

    static final long TOTAL_CAP = 200;
    static final double TOTAL_RATE = 2000;
    static final long OPS = 8_000_000L;
    static final int REPEAT = 5;

    public static void main(String[] args) throws Exception {
        System.out.println("=== A. 单桶，加线程（有竞争）===");
        for (int th : new int[]{1, 2, 4, 8, 16, 32}) {
            long med = medianQps(1, th);
            System.out.printf(Locale.ROOT, "  shards=1 threads=%2d  median_qps=%,d%n", th, med);
        }

        System.out.println("=== B. 分片=线程（每线程独占一桶，总配额均分）===");
        long best = 0;
        int bestN = 0;
        for (int n : new int[]{1, 2, 4, 8, 16, 32, 64}) {
            long med = medianQps(n, n);
            System.out.printf(Locale.ROOT, "  shards=threads=%2d  median_qps=%,d%n", n, med);
            if (med > best) { best = med; bestN = n; }
        }

        System.out.println("=== C. 分片固定，线程加倍（多线程抢同一分片）===");
        for (int shards : new int[]{4, 8, 16}) {
            for (int th : new int[]{shards, shards * 2, shards * 4}) {
                long med = medianQps(shards, th);
                System.out.printf(Locale.ROOT, "  shards=%2d threads=%2d  median_qps=%,d%n", shards, th, med);
            }
        }

        System.out.printf(Locale.ROOT,
                "%n结论候选: shards=threads=%d  median=%,d tryAcquire/s%n", bestN, best);
    }

    static long medianQps(int shards, int threads) throws InterruptedException {
        long[] xs = new long[REPEAT];
        for (int i = 0; i < REPEAT; i++) xs[i] = runOnce(shards, threads);
        Arrays.sort(xs);
        return xs[REPEAT / 2];
    }

    static long runOnce(int shards, int threads) throws InterruptedException {
        long cap = Math.max(1, TOTAL_CAP / shards);
        double rate = TOTAL_RATE / shards;
        TokenBucket[] buckets = new TokenBucket[shards];
        for (int i = 0; i < shards; i++) buckets[i] = new TokenBucket(cap, rate);

        long per = OPS / threads;
        CountDownLatch gate = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(threads);
        LongAdder finished = new LongAdder();

        for (int t = 0; t < threads; t++) {
            final int tid = t;
            new Thread(() -> {
                try {
                    gate.await();
                    TokenBucket b = buckets[tid % shards];
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

    private RateLimitTune2() {}
}
