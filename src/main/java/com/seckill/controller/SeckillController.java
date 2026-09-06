package com.seckill.controller;

import com.seckill.seckill.PayService;
import com.seckill.seckill.SeckillService;
import com.seckill.cache.VoucherCacheService;
import com.seckill.seckill.VoucherService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

/**
 * 美食汇券接口（用户 × 美食优惠券）：
 * - GET  /voucher/{id}/stock        查余量（读）
 * - POST /voucher/{id}/seckill      抢券（写，一人一券一次）
 * - POST /voucher/{id}/addStock     加库存（写操作增加券量）
 */
@RestController
@RequestMapping("/voucher")
public class SeckillController {

    private final SeckillService seckillService;
    private final PayService payService;
    private final VoucherService voucherService;
    private final VoucherCacheService voucherCacheService;

    public SeckillController(SeckillService seckillService, PayService payService,
                             VoucherService voucherService, VoucherCacheService voucherCacheService) {
        this.seckillService = seckillService;
        this.payService = payService;
        this.voucherService = voucherService;
        this.voucherCacheService = voucherCacheService;
    }

    /** 券详情（多级缓存：Caffeine L1 → Redis 从库 L2 → MySQL，布隆防穿透 + 互斥锁防击穿） */
    @GetMapping("/{voucherId}/detail")
    public Map<String, Object> detail(@PathVariable long voucherId) {
        String d = voucherCacheService.queryDetail(voucherId);
        return Map.of("code", d == null ? 404 : 0, "data", d == null ? "voucher not found" : d);
    }

    /** 券余量（布隆防穿透 + Redis 主库直读） */
    @GetMapping("/{voucherId}/stock")
    public Map<String, Object> stock(@PathVariable long voucherId) {
        Long stock = voucherService.queryStock(voucherId);
        return Map.of("code", stock == null ? 404 : 0, "data", stock == null ? "voucher not found" : stock);
    }

    /** 秒杀下单：Lua + 同事务待支付订单；SUCCESS 时已可支付 */
    @PostMapping("/{voucherId}/seckill")
    public Map<String, Object> seckill(@PathVariable long voucherId, @RequestParam long userId) {
        SeckillService.Outcome o = seckillService.seckill(voucherId, userId);
        if (o.result() == SeckillService.Result.SUCCESS && o.orderNo() != null) {
            return Map.of("code", o.result().ordinal(), "msg", o.result().name(), "orderNo", o.orderNo());
        }
        return Map.of("code", o.result().ordinal(), "msg", o.result().name());
    }

    /** 更新券详情（先更 DB → 删 L2 → pub/sub 广播踢全部实例 L1） */
    @PostMapping("/{voucherId}/detail")
    public Map<String, Object> updateDetail(@PathVariable long voucherId,
                                            @RequestParam String name,
                                            @RequestParam String description) {
        boolean ok = voucherCacheService.updateDetail(voucherId, name, description);
        return Map.of("code", ok ? 0 : 404, "msg", ok ? "updated" : "voucher not found");
    }

    /** 加库存 */
    @PostMapping("/{voucherId}/addStock")
    public Map<String, Object> addStock(@PathVariable long voucherId, @RequestParam int amount) {
        long stock = voucherService.addStock(voucherId, amount);
        return Map.of("code", stock < 0 ? 404 : 0, "msg", stock < 0 ? "voucher not found" : "stock=" + stock);
    }

    /** 支付订单（幂等）：已支付再调仍返回 paid；成功后 Kafka 异步加积分 */
    @PostMapping("/order/{voucherId}/{userId}/pay")
    public Map<String, Object> pay(@PathVariable long voucherId, @PathVariable long userId) {
        PayService.PayResult r = payService.pay(voucherId, userId);
        return Map.of("code", r.ordinal(), "msg", switch (r) {
            case OK -> "paid";
            case ORDER_NOT_PAYABLE -> "order not payable";
            case VOUCHER_MISSING -> "voucher missing";
            case INSUFFICIENT_BALANCE -> "insufficient balance";
        });
    }

    /** 用户取消待支付订单 */
    @PostMapping("/order/{voucherId}/{userId}/cancel")
    public Map<String, Object> cancel(@PathVariable long voucherId, @PathVariable long userId) {
        PayService.CancelResult r = payService.cancelByUser(voucherId, userId);
        return Map.of("code", r.ordinal(), "msg", switch (r) {
            case OK -> "cancelled";
            case ORDER_NOT_CANCELABLE -> "order not cancelable";
        });
    }
}
