package com.seckill.controller;

import com.seckill.seckill.DualRateLimiter;
import com.seckill.seckill.InventoryReconciler;
import com.seckill.seckill.SeckillService;
import com.seckill.wallet.UserWalletService;
import org.apache.kafka.clients.producer.Producer;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 压测辅助接口（仅演示环境，生产必须鉴权/隔离）：
 * - reset：一轮压测前把库存/订单/延迟队列/积分恢复初始态
 * - audit：三段式库存交叉对账（redis↔db available、db locked↔待支付订单、db sold↔已支付订单）
 */
@RestController
@RequestMapping("/admin")
public class AdminController {

    private final JdbcTemplate jdbc;
    private final SeckillService seckillService;
    private final Producer<Long, String> producer;
    @Value("${seckill.kafka.topic}") private String topic;
    private final int voucherCount;

    private final InventoryReconciler reconciler;
    private final UserWalletService walletService;

    public AdminController(JdbcTemplate jdbc, SeckillService seckillService,
                           Producer<Long, String> producer,
                           InventoryReconciler reconciler,
                           UserWalletService walletService,
                           @Value("${seckill.voucher-count}") int voucherCount) {
        this.jdbc = jdbc;
        this.seckillService = seckillService;
        this.producer = producer;
        this.reconciler = reconciler;
        this.walletService = walletService;
        this.voucherCount = voucherCount;
    }

    /** 演示用：向积分主题注入毒消息，验证 DLT */
    @PostMapping("/poison")
    public Map<String, Object> poison() throws Exception {
        producer.send(new ProducerRecord<>(topic, "poison:not-a-number:xxx")).get();
        return Map.of("code", 0, "msg", "poison injected, watch DLT topic " + topic + "-dlt");
    }

    /** 手动触发 Redis 秒杀残缺态对账补偿（stock+users 不变量） */
    @PostMapping("/reconcile")
    public Map<String, Object> reconcile() {
        Map<String, Object> r = reconciler.reconcileAll();
        r.put("code", 0);
        return r;
    }

    /** 恢复初始态：三段库存重置、清订单、清延迟队列与 Redis 库存、积分归零 */
    @PostMapping("/reset")
    public Map<String, Object> reset(@RequestParam(defaultValue = "1000") int stock) {
        jdbc.update("UPDATE tb_seckill_voucher SET available = ?, locked = 0, sold = 0, version = version + 1", stock);
        jdbc.execute("TRUNCATE tb_voucher_order");
        jdbc.update("UPDATE tb_shop SET balance = 0");
        jdbc.update("UPDATE tb_user SET balance = 10000, points = 0");
        walletService.clearAllBalanceCache();
        for (long v = 1; v <= voucherCount; v++) {
            seckillService.resetRedis(v, stock);
        }
        seckillService.clearUnpaidQueue();
        seckillService.nextEpoch();
        seckillService.rateLimiter.resetStats();
        return Map.of("code", 0, "msg", "reset to stock=" + stock);
    }

    /** 切换限流：token|leaky；可选 capacity/ratePerSec/shards */
    @PostMapping("/rate")
    public Map<String, Object> rate(@RequestParam String mode,
                                    @RequestParam(required = false) Long capacity,
                                    @RequestParam(required = false) Double ratePerSec,
                                    @RequestParam(required = false) Integer shards) {
        DualRateLimiter.Mode m;
        if ("leaky".equalsIgnoreCase(mode)) m = DualRateLimiter.Mode.LEAKY;
        else if ("token".equalsIgnoreCase(mode)) m = DualRateLimiter.Mode.TOKEN;
        else return Map.of("code", 1, "msg", "mode must be token|leaky");
        DualRateLimiter lim = seckillService.rateLimiter;
        lim.setMode(m);
        if (capacity != null && ratePerSec != null) lim.reconfigure(capacity, ratePerSec);
        if (shards != null) lim.setShardCount(shards);
        lim.resetStats();
        return rateStats();
    }

    /** 限流统计：算法/分片 + 通过/拒绝 */
    @GetMapping("/rate")
    public Map<String, Object> rateStats() {
        DualRateLimiter lim = seckillService.rateLimiter;
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("code", 0);
        out.put("mode", lim.name());
        out.put("shards", lim.shardCount());
        out.put("capacity", lim.totalCapacity());
        out.put("ratePerSec", lim.totalRate());
        out.put("passed", lim.passed());
        out.put("rejected", lim.rejected());
        return out;
    }

}
