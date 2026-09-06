package com.seckill.bench;

import com.seckill.seckill.DualRateLimiter;
import com.seckill.seckill.RateLimiter;

import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 独立限流微压测（无需 Spring / 中间件）：
 * java -cp target/classes com.seckill.bench.RateLimitBench [threads=8] [ops=2000000]
 */
public final class RateLimitBench {

    public static void main(String[] args) throws Exception {
        int threads = args.length > 0 ? Integer.parseInt(args[0]) : 8;
        long ops = args.length > 1 ? Long.parseLong(args[1]) : 2_000_000L;
        long capacity = 200;
        double rate = 2000;

        System.out.printf(Locale.ROOT,
                "限流微压测: capacity=%d rate=%.0f/s threads=%d ops=%d%n",
                capacity, rate, threads, ops);

        for (DualRateLimiter.Mode mode : DualRateLimiter.Mode.values()) {
            DualRateLimiter lim = new DualRateLimiter(capacity, rate, mode);
            long qps = burstQps(lim, threads, ops);
            System.out.printf(Locale.ROOT,
                    "  [%s] 判定吞吐=%,d tryAcquire/s  通过=%d 拒绝=%d%n",
                    mode.name().toLowerCase(), qps, lim.passed(), lim.rejected());
        }

        // 持续 3s：观察稳态允许速率（应贴近 rate）
        System.out.println("稳态放行（单线程持续 3s，期望≈2000/s）：");
        for (DualRateLimiter.Mode mode : DualRateLimiter.Mode.values()) {
            DualRateLimiter lim = new DualRateLimiter(capacity, rate, mode);
            long end = System.nanoTime() + 3_000_000_000L;
            while (System.nanoTime() < end) lim.tryAcquire();
            System.out.printf(Locale.ROOT,
                    "  [%s] 放行=%d (≈%.0f/s) 拒绝=%d%n",
                    mode.name().toLowerCase(), lim.passed(), lim.passed() / 3.0, lim.rejected());
        }
    }

    static long burstQps(RateLimiter lim, int threads, long totalOps) throws InterruptedException {
        long per = totalOps / threads;
        CountDownLatch startGate = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(threads);
        AtomicLong finished = new AtomicLong();

        for (int t = 0; t < threads; t++) {
            new Thread(() -> {
                try {
                    startGate.await();
                    for (long i = 0; i < per; i++) lim.tryAcquire();
                    finished.addAndGet(per);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    done.countDown();
                }
            }).start();
        }

        long t0 = System.nanoTime();
        startGate.countDown();
        done.await();
        double secs = (System.nanoTime() - t0) / 1e9;
        return Math.round(finished.get() / secs);
    }

    private RateLimitBench() {}
}
