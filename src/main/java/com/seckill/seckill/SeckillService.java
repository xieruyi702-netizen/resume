package com.seckill.seckill;

import com.seckill.mapper.OrderMapper;
import com.seckill.mapper.VoucherMapper;
import com.seckill.util.SnowflakeIdGen;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;

/**
 * 美食汇优惠券秒杀：限流 → Redis Lua → 同事务落待支付订单 + lockStock → 挂延迟队列。
 * 返回 SUCCESS 时订单已可支付；积分发放改在支付成功后走 Kafka。
 */
@Service
public class SeckillService {

    static final String LUA_SECKILL = """
            if redis.call('SISMEMBER', KEYS[2], ARGV[1]) == 1 then
                return 0
            end
            local stock = tonumber(redis.call('GET', KEYS[1]))
            if stock == nil or stock <= 0 then
                return 0
            end
            redis.call('SADD', KEYS[2], ARGV[1])
            local ok = redis.pcall('DECR', KEYS[1])
            if type(ok) == 'table' and ok['err'] then
                redis.call('SREM', KEYS[2], ARGV[1])
                return redis.error_reply(ok['err'])
            end
            return 1
            """;

    static final String LUA_ROLLBACK = """
            if redis.call('SREM', KEYS[2], ARGV[1]) == 1 then
                local ok = redis.pcall('INCR', KEYS[1])
                if type(ok) == 'table' and ok['err'] then
                    redis.call('SADD', KEYS[2], ARGV[1])
                    return redis.error_reply(ok['err'])
                end
                return 1
            end
            return 0
            """;

    private final JedisPool master;
    private final VoucherMapper voucherMapper;
    private final OrderMapper orderMapper;
    private final TransactionTemplate tx;
    private final SnowflakeIdGen snowflake;
    private final long orderTimeoutMs;
    public final DualRateLimiter rateLimiter;

    public SeckillService(@org.springframework.beans.factory.annotation.Qualifier("masterPool") JedisPool master,
                          VoucherMapper voucherMapper,
                          OrderMapper orderMapper,
                          PlatformTransactionManager ptm,
                          @Value("${seckill.order-timeout-ms}") long orderTimeoutMs,
                          @Value("${seckill.snowflake.worker-id}") long workerId,
                          @Value("${seckill.snowflake.datacenter-id}") long datacenterId,
                          @Value("${seckill.rate.capacity}") long capacity,
                          @Value("${seckill.rate.refill-per-sec}") double refillPerSec,
                          @Value("${seckill.rate.mode:token}") String rateMode,
                          @Value("${seckill.rate.shards:1}") int shards) {
        this.master = master;
        this.voucherMapper = voucherMapper;
        this.orderMapper = orderMapper;
        this.tx = new TransactionTemplate(ptm);
        this.orderTimeoutMs = orderTimeoutMs;
        this.snowflake = new SnowflakeIdGen(workerId, datacenterId);
        DualRateLimiter.Mode mode = "leaky".equalsIgnoreCase(rateMode)
                ? DualRateLimiter.Mode.LEAKY : DualRateLimiter.Mode.TOKEN;
        this.rateLimiter = new DualRateLimiter(capacity, refillPerSec, mode, shards);
    }

    public enum Result { SUCCESS, RATE_LIMITED, SOLD_OUT, DUPLICATED, DB_ERROR }

    public record Outcome(Result result, Long orderNo) {
        static Outcome of(Result r) { return new Outcome(r, null); }
        static Outcome ok(long orderNo) { return new Outcome(Result.SUCCESS, orderNo); }
    }

    /** 秒杀入口：限流 → Lua → 同事务建待支付单 → 延迟队列 */
    public Outcome seckill(long voucherId, long userId) {
        if (!rateLimiter.tryAcquire()) return Outcome.of(Result.RATE_LIMITED);

        boolean luaOk;
        try (Jedis j = master.getResource()) {
            Object r = j.eval(LUA_SECKILL, 2,
                    "seckill:stock:" + voucherId, "seckill:users:" + voucherId, String.valueOf(userId));
            luaOk = ((Long) r) == 1;
        } catch (Exception e) {
            try { rollback(voucherId, userId); } catch (Exception ignored) {}
            return Outcome.of(Result.DB_ERROR);
        }
        if (!luaOk) {
            return Outcome.of(Result.SOLD_OUT);
        }

        try {
            long orderNo = createUnpaidOrder(voucherId, userId);
            return Outcome.ok(orderNo);
        } catch (Exception e) {
            rollback(voucherId, userId);
            return Outcome.of(Result.DB_ERROR);
        }
    }

    private final SnowflakeIdGen dbSnowflake = new SnowflakeIdGen(31, 1);

    public boolean seckillDb(long voucherId, long userId) {
        if (voucherMapper.deductStock(voucherId) == 0) return false;
        try {
            orderMapper.insertUnpaid(dbSnowflake.nextId(), voucherId, userId);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    public long nextEpoch() {
        long now = System.currentTimeMillis();
        try (Jedis j = master.getResource()) {
            j.set("seckill:epoch", String.valueOf(now));
        }
        return now;
    }

    /** 同步落单：待支付订单 + lockStock；提交后挂 ZSet */
    long createUnpaidOrder(long voucherId, long userId) {
        long orderNo = snowflake.nextId();
        String member = voucherId + ":" + userId;
        final long[] resolved = { orderNo };

        tx.executeWithoutResult(status -> {
            int inserted = orderMapper.insertUnpaid(orderNo, voucherId, userId);
            if (inserted == 0) {
                Long existing = orderMapper.selectOrderNo(voucherId, userId);
                if (existing == null) {
                    throw new IllegalStateException("order missing after duplicate insert");
                }
                resolved[0] = existing;
                return;
            }
            if (voucherMapper.lockStock(voucherId) == 0) {
                throw new IllegalStateException("lockStock failed voucher=" + voucherId);
            }
        });

        try (Jedis j = master.getResource()) {
            if (j.zscore("orders:unpaid", member) == null) {
                j.zadd("orders:unpaid", System.currentTimeMillis() + orderTimeoutMs, member);
            }
        }
        return resolved[0];
    }

    public void resetRedis(long voucherId, int stock) {
        try (Jedis j = master.getResource()) {
            j.set("seckill:stock:" + voucherId, String.valueOf(stock));
            j.del("seckill:users:" + voucherId);
        }
    }

    public void clearUnpaidQueue() {
        try (Jedis j = master.getResource()) {
            j.del("orders:unpaid");
        }
    }

    public long redisStock(long voucherId) {
        try (Jedis j = master.getResource()) {
            String s = j.get("seckill:stock:" + voucherId);
            return s == null ? 0 : Long.parseLong(s);
        }
    }

    public long redisUserCount(long voucherId) {
        try (Jedis j = master.getResource()) {
            Long n = j.scard("seckill:users:" + voucherId);
            return n == null ? 0 : n;
        }
    }

    public java.util.Set<String> redisUsers(long voucherId) {
        try (Jedis j = master.getResource()) {
            java.util.Set<String> s = j.smembers("seckill:users:" + voucherId);
            return s == null ? java.util.Set.of() : s;
        }
    }

    public long compensateOrphanDecr(long voucherId, long delta) {
        if (delta <= 0) return 0;
        try (Jedis j = master.getResource()) {
            Long after = null;
            for (long i = 0; i < delta; i++) {
                after = j.incr("seckill:stock:" + voucherId);
            }
            return after == null ? 0 : after;
        }
    }

    public boolean compensateGhostUser(long voucherId, long userId) {
        try (Jedis j = master.getResource()) {
            Long n = j.srem("seckill:users:" + voucherId, String.valueOf(userId));
            return n != null && n > 0;
        }
    }

    public boolean rollback(long voucherId, long userId) {
        try (Jedis j = master.getResource()) {
            Object r = j.eval(LUA_ROLLBACK, 2,
                    "seckill:stock:" + voucherId, "seckill:users:" + voucherId, String.valueOf(userId));
            return ((Long) r) == 1;
        }
    }
}
