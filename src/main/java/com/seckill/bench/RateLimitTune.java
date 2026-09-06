package com.seckill.bench;

import com.seckill.seckill.TokenBucket;

import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.LongAdder;

/**
 * 扫参：分片数 × 线程数 × 路由策略，找 tryAcquire 判定吞吐峰值。
 * 总配额固定：capacity=200, rate=2000/s，均分到各分片。
 *
 * java -cp target/classes com.seckill.bench.RateLimitTune
 */
public final class RateLimitTune {

    static final long TOTAL_CAP = 200;
    static final double TOTAL_RATE = 2000;
    static final long OPS = 4_000_000L;

    enum Route { AFFINITY, LOCAL_RR }

    public static void main(String[] args) throws Exception {
        int[] shardsList = {1, 2, 4, 8, 16, 32, 64};
        int[] threadsList = {1, 2, 4, 8, 16, 32, 64};

        System.out.printf(Locale.ROOT,
                "扫参: totalCap=%d totalRate=%.0f/s ops=%d%n", TOTAL_CAP, TOTAL_RATE, OPS);
        System.out.printf(Locale.ROOT, "%8s %6s %8s %14s %10s%n",
                "route", "shards", "threads", "qps", "passed");

        long bestQps = 0;
        String best = "";

        for (Route route : Route.values()) {
            for (int shards : shardsList) {
                for (int threads : threadsList) {
                    Result r = run(shards, threads, OPS, route);
                    System.out.printf(Locale.ROOT, "%8s %6d %8d %,14d %10d%n",
                            route, shards, threads, r.qps, r.passed);
                    if (r.qps > bestQps) {
                        bestQps = r.qps;
                        best = route + " shards=" + shards + " threads=" + threads;
                    }
                }
            }
        }

        Result base = run(1, 1, OPS, Route.AFFINITY);
        System.out.printf(Locale.ROOT,
                "%n最佳: %s  qps=%,d/s%n基线单桶单线程: %,d/s  (%.2fx)%n",
                best, bestQps, base.qps, bestQps / (double) base.qps);
    }

    static Result run(int shards, int threads, long totalOps, Route route) throws InterruptedException {
        long cap = Math.max(1, TOTAL_CAP / shards);
        double rate = TOTAL_RATE / shards;
        TokenBucket[] buckets = new TokenBucket[shards];
        for (int i = 0; i < shards; i++) buckets[i] = new TokenBucket(cap, rate);

        long per = totalOps / threads;
        CountDownLatch gate = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(threads);
        LongAdder finished = new LongAdder();

        for (int t = 0; t < threads; t++) {
            final int tid = t;
            new Thread(() -> {
                try {
                    gate.await();
                    if (route == Route.AFFINITY) {
                        TokenBucket b = buckets[tid % shards];
                        for (long i = 0; i < per; i++) b.tryAcquire();
                    } else {
                        int idx = tid % shards;
                        for (long i = 0; i < per; i++) {
                            buckets[idx].tryAcquire();
                            idx++;
                            if (idx == shards) idx = 0;
                        }
                    }
                    finished.add(per);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    done.countDown();
                }
            }, "rl-" + tid).start();
        }

        long t0 = System.nanoTime();
        gate.countDown();
        done.await();
        double secs = (System.nanoTime() - t0) / 1e9;
        long qps = Math.round(finished.sum() / secs);

        long passed = 0;
        for (TokenBucket b : buckets) passed += b.passed();
        return new Result(qps, passed);
    }

    record Result(long qps, long passed) {}

    private RateLimitTune() {}
}
