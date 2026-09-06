package com.seckill.wallet;

import com.seckill.mapper.UserMapper;
import org.redisson.api.RAtomicLong;
import org.redisson.api.RKeys;
import org.redisson.api.RLock;
import org.redisson.api.RedissonClient;
import org.springframework.stereotype.Service;

import java.util.concurrent.TimeUnit;

/**
 * 用户余额：真相在 MySQL；热路径读走 Redis（Redisson RAtomicLong）。
 * 写路径 Cache Aside：锁内改 MySQL 后淘汰 Redis（不回写），下次读 miss 再从 DB 加载。
 * unlock 由 Redisson 看门狗续期 + Lua 原子释放（持锁线程校验）。
 */
@Service
public class UserWalletService {

    private static final String BAL_KEY = "wallet:user:";
    private static final String LOCK_KEY = "lock:wallet:";

    private final RedissonClient redisson;
    private final UserMapper userMapper;

    public UserWalletService(RedissonClient redisson, UserMapper userMapper) {
        this.redisson = redisson;
        this.userMapper = userMapper;
    }

    private static String balKey(long userId) { return BAL_KEY + userId; }
    private static String lockKey(long userId) { return LOCK_KEY + userId; }

    /**
     * 查余额：先 Redis；未命中则加锁加载 MySQL 并写入 Redis。
     * @return null = 用户不存在
     */
    public Long getBalance(long userId) {
        RAtomicLong atomic = redisson.getAtomicLong(balKey(userId));
        if (atomic.isExists()) {
            return atomic.get();
        }
        RLock lock = redisson.getLock(lockKey(userId));
        try {
            if (!lock.tryLock(2, 5, TimeUnit.SECONDS)) {
                // 抢锁失败：再读一次缓存，否则直接查库（不写缓存，避免无锁击穿写）
                if (atomic.isExists()) return atomic.get();
                return userMapper.selectBalance(userId);
            }
            try {
                if (atomic.isExists()) return atomic.get();
                Long db = userMapper.selectBalance(userId);
                if (db == null) return null;
                atomic.set(db);
                return db;
            } finally {
                if (lock.isHeldByCurrentThread()) lock.unlock();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return userMapper.selectBalance(userId);
        }
    }

    public Long getPoints(long userId) {
        return userMapper.selectPoints(userId);
    }

    /** 充值：锁内改 MySQL，成功后删 Redis（不回写） */
    public Long recharge(long userId, long amount) {
        if (amount <= 0) throw new IllegalArgumentException("amount must be > 0");
        RLock lock = redisson.getLock(lockKey(userId));
        try {
            if (!lock.tryLock(3, 10, TimeUnit.SECONDS)) {
                throw new IllegalStateException("wallet busy");
            }
            try {
                userMapper.ensure(userId, 0);
                userMapper.credit(userId, amount);
                Long bal = userMapper.selectBalance(userId);
                invalidate(userId);
                return bal;
            } finally {
                if (lock.isHeldByCurrentThread()) lock.unlock();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted", e);
        }
    }

    /**
     * 支付扣减：锁内可选 Redis 快失败 → MySQL CAS 扣减 → 删 Redis（不回写）。
     * @return false = 余额不足或用户不存在
     */
    public boolean deduct(long userId, long amount) {
        RLock lock = redisson.getLock(lockKey(userId));
        try {
            if (!lock.tryLock(3, 10, TimeUnit.SECONDS)) {
                return false;
            }
            try {
                RAtomicLong atomic = redisson.getAtomicLong(balKey(userId));
                if (atomic.isExists() && atomic.get() < amount) {
                    return false;
                }
                if (userMapper.deduct(userId, amount) == 0) {
                    // DB 不足：淘汰可能偏高的缓存
                    invalidate(userId);
                    return false;
                }
                invalidate(userId);
                return true;
            } finally {
                if (lock.isHeldByCurrentThread()) lock.unlock();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    /** 写后淘汰缓存（与 Cache Aside 一致） */
    public void invalidate(long userId) {
        redisson.getAtomicLong(balKey(userId)).delete();
    }

    /** 压测 reset：清掉所有用户余额缓存 */
    public long clearAllBalanceCache() {
        RKeys keys = redisson.getKeys();
        return keys.deleteByPattern(BAL_KEY + "*");
    }
}
