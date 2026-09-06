package com.seckill.seckill;

import com.seckill.mapper.OrderMapper;
import com.seckill.mapper.ShopMapper;
import com.seckill.mapper.VoucherDetailMapper;
import com.seckill.mapper.VoucherMapper;
import com.seckill.wallet.UserWalletService;
import org.apache.kafka.clients.producer.Producer;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;

import java.util.List;
import java.util.Map;

/**
 * 支付与取消：
 * - pay：扣余额 → 商铺入账/订单已支付/库存 → ZREM → Kafka 异步加积分
 * - 超时取消 status=2；用户取消 status=3
 */
@Service
public class PayService {

    public enum PayResult {
        OK, ORDER_NOT_PAYABLE, VOUCHER_MISSING, INSUFFICIENT_BALANCE
    }

    public enum CancelResult {
        OK, ORDER_NOT_CANCELABLE
    }

    private final OrderMapper orderMapper;
    private final VoucherMapper voucherMapper;
    private final VoucherDetailMapper detailMapper;
    private final ShopMapper shopMapper;
    private final UserWalletService walletService;
    private final SeckillService seckillService;
    private final JedisPool master;
    private final TransactionTemplate tx;
    private final Producer<Long, String> producer;
    private final String pointsTopic;

    public PayService(OrderMapper orderMapper, VoucherMapper voucherMapper,
                      VoucherDetailMapper detailMapper, ShopMapper shopMapper,
                      UserWalletService walletService, SeckillService seckillService,
                      PlatformTransactionManager ptm,
                      @org.springframework.beans.factory.annotation.Qualifier("masterPool") JedisPool master,
                      Producer<Long, String> producer,
                      @Value("${seckill.kafka.topic}") String pointsTopic) {
        this.orderMapper = orderMapper;
        this.voucherMapper = voucherMapper;
        this.detailMapper = detailMapper;
        this.shopMapper = shopMapper;
        this.walletService = walletService;
        this.seckillService = seckillService;
        this.master = master;
        this.producer = producer;
        this.pointsTopic = pointsTopic;
        this.tx = new TransactionTemplate(ptm);
    }

    /**
     * 支付幂等：已支付再调返回 OK；仅待支付走扣款入账。
     * 并发：CAS 落败若发现已是已支付，退回本次误扣后仍返回 OK。
     */
    public PayResult pay(long voucherId, long userId) {
        Integer status = orderMapper.selectStatus(voucherId, userId);
        if (status == null) {
            return PayResult.ORDER_NOT_PAYABLE;
        }
        if (status == 1) {
            return PayResult.OK; // 已支付：幂等成功，不重复扣款/入账/发积分
        }
        if (status != 0) {
            return PayResult.ORDER_NOT_PAYABLE; // 超时取消 / 用户取消
        }

        Map<String, Object> info = detailMapper.selectPayInfo(voucherId);
        if (info == null) {
            return PayResult.VOUCHER_MISSING;
        }
        long shopId = ((Number) info.get("shopId")).longValue();
        int price = ((Number) info.get("price")).intValue();

        if (!walletService.deduct(userId, price)) {
            return PayResult.INSUFFICIENT_BALANCE;
        }
        final long[] orderNoHolder = {0};
        try {
            tx.executeWithoutResult(statusObj -> {
                if (shopMapper.credit(shopId, price) == 0) {
                    throw new IllegalStateException("shop not found: " + shopId);
                }
                if (orderMapper.pay(voucherId, userId, price, shopId) == 0) {
                    throw new IllegalStateException("order pay race");
                }
                voucherMapper.payConvert(voucherId);
                Long ono = orderMapper.selectOrderNo(voucherId, userId);
                orderNoHolder[0] = ono == null ? 0 : ono;
            });
        } catch (RuntimeException e) {
            walletService.recharge(userId, price); // 退回本请求误扣
            if (e.getMessage() != null && e.getMessage().contains("order pay race")) {
                Integer after = orderMapper.selectStatus(voucherId, userId);
                if (after != null && after == 1) {
                    return PayResult.OK; // 并发另一请求已付成功 → 幂等 OK
                }
                return PayResult.ORDER_NOT_PAYABLE;
            }
            throw e;
        }

        try (Jedis j = master.getResource()) {
            j.zrem("orders:unpaid", voucherId + ":" + userId);
        }
        publishPoints(orderNoHolder[0], userId, price);
        return PayResult.OK;
    }

    /** 消息体：orderNo:userId:points（积分=支付金额分，消费端幂等） */
    public void publishPoints(long orderNo, long userId, int points) {
        if (orderNo <= 0 || points <= 0) return;
        String body = orderNo + ":" + userId + ":" + points;
        producer.send(new ProducerRecord<>(pointsTopic, userId, body));
    }

    /** 用户主动取消待支付单 */
    public CancelResult cancelByUser(long voucherId, long userId) {
        int updated = orderMapper.cancelByUser(voucherId, userId);
        if (updated == 0) {
            return CancelResult.ORDER_NOT_CANCELABLE;
        }
        voucherMapper.cancelRestore(voucherId);
        seckillService.rollback(voucherId, userId);
        try (Jedis j = master.getResource()) {
            j.zrem("orders:unpaid", voucherId + ":" + userId);
        }
        return CancelResult.OK;
    }

    public int cancelExpired(int batchSize) {
        long now = System.currentTimeMillis();
        List<String> due;
        try (Jedis j = master.getResource()) {
            due = j.zrangeByScore("orders:unpaid", 0, now, 0, batchSize);
        }
        int cancelled = 0;
        for (String member : due) {
            String[] parts = member.split(":");
            long voucherId = Long.parseLong(parts[0]);
            long userId = Long.parseLong(parts[1]);

            int updated = orderMapper.cancelTimeout(voucherId, userId);
            if (updated > 0) {
                voucherMapper.cancelRestore(voucherId);
                seckillService.rollback(voucherId, userId);
                cancelled++;
            }
            try (Jedis j = master.getResource()) {
                j.zrem("orders:unpaid", member);
            }
        }
        return cancelled;
    }

    public long unpaidQueueSize() {
        try (Jedis j = master.getResource()) {
            return j.zcard("orders:unpaid");
        }
    }
}
