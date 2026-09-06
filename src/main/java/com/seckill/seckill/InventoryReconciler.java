package com.seckill.seckill;

import com.seckill.mapper.OrderMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Redis 秒杀残缺态定时补偿。
 * 不变量：{@code redis_stock + SCARD(users) == N}。
 * 和偏大时移出无生效订单（待支付/已支付）的幽灵用户。
 */
@Component
public class InventoryReconciler {

    private final SeckillService seckill;
    private final OrderMapper orderMapper;
    private final int voucherCount;
    private final long initialStock;
    private final boolean enabled;

    public InventoryReconciler(SeckillService seckill,
                               OrderMapper orderMapper,
                               @Value("${seckill.voucher-count:1000}") int voucherCount,
                               @Value("${seckill.reconcile.initial-stock:1000}") long initialStock,
                               @Value("${seckill.reconcile.enabled:true}") boolean enabled) {
        this.seckill = seckill;
        this.orderMapper = orderMapper;
        this.voucherCount = voucherCount;
        this.initialStock = initialStock;
        this.enabled = enabled;
    }

    @Scheduled(fixedDelayString = "${seckill.reconcile.delay-ms:5000}")
    public void scheduled() {
        if (!enabled) return;
        reconcileAll();
    }

    public Map<String, Object> reconcileAll() {
        long fixedDecr = 0, fixedGhost = 0, checked = 0, broken = 0;
        for (long v = 1; v <= voucherCount; v++) {
            Map<String, Long> r = reconcileOne(v);
            checked++;
            fixedDecr += r.getOrDefault("fixedOrphanDecr", 0L);
            fixedGhost += r.getOrDefault("fixedGhostUsers", 0L);
            if (r.getOrDefault("brokenBefore", 0L) == 1L) broken++;
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("checked", checked);
        out.put("brokenBefore", broken);
        out.put("fixedOrphanDecr", fixedDecr);
        out.put("fixedGhostUsers", fixedGhost);
        out.put("initialStock", initialStock);
        return out;
    }

    public Map<String, Long> reconcileOne(long voucherId) {
        Map<String, Long> out = new LinkedHashMap<>();
        long stock = seckill.redisStock(voucherId);
        long users = seckill.redisUserCount(voucherId);
        long sum = stock + users;
        out.put("stock", stock);
        out.put("users", users);
        out.put("sum", sum);
        out.put("brokenBefore", sum == initialStock ? 0L : 1L);

        long fixedDecr = 0;
        long fixedGhost = 0;

        if (sum < initialStock) {
            long delta = initialStock - sum;
            seckill.compensateOrphanDecr(voucherId, delta);
            fixedDecr = delta;
        } else if (sum > initialStock) {
            Set<String> members = seckill.redisUsers(voucherId);
            for (String uidStr : members) {
                if (stock + seckill.redisUserCount(voucherId) <= initialStock) break;
                long userId;
                try {
                    userId = Long.parseLong(uidStr);
                } catch (NumberFormatException e) {
                    continue;
                }
                Integer status = orderMapper.selectStatus(voucherId, userId);
                boolean activeOrder = status != null && (status == 0 || status == 1);
                if (!activeOrder) {
                    if (seckill.compensateGhostUser(voucherId, userId)) {
                        fixedGhost++;
                    }
                }
            }
            stock = seckill.redisStock(voucherId);
            users = seckill.redisUserCount(voucherId);
            out.put("sumAfterGhost", stock + users);
        }

        out.put("fixedOrphanDecr", fixedDecr);
        out.put("fixedGhostUsers", fixedGhost);
        out.put("sumAfter", seckill.redisStock(voucherId) + seckill.redisUserCount(voucherId));
        return out;
    }
}
