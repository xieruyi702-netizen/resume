package com.seckill.controller;

import com.seckill.mapper.ShopMapper;
import com.seckill.wallet.UserWalletService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

/** 用户钱包（读：Redis Cache Aside；写：MySQL 后删缓存）/ 商铺账户（仍 MySQL）。金额单位：分。 */
@RestController
@RequestMapping
public class WalletController {

    private final UserWalletService walletService;
    private final ShopMapper shopMapper;

    public WalletController(UserWalletService walletService, ShopMapper shopMapper) {
        this.walletService = walletService;
        this.shopMapper = shopMapper;
    }

    @GetMapping("/user/{userId}/balance")
    public Map<String, Object> userBalance(@PathVariable long userId) {
        Long bal = walletService.getBalance(userId);
        if (bal == null) {
            return Map.of("code", 404, "msg", "user not found");
        }
        Long points = walletService.getPoints(userId);
        return Map.of("code", 0, "userId", userId, "balance", bal,
                "points", points == null ? 0 : points, "source", "redis-or-db");
    }

    /** 充值：Redisson 锁内写 MySQL，再淘汰 Redis */
    @PostMapping("/user/{userId}/recharge")
    public Map<String, Object> recharge(@PathVariable long userId, @RequestParam long amount) {
        if (amount <= 0) {
            return Map.of("code", 1, "msg", "amount must be > 0");
        }
        try {
            Long bal = walletService.recharge(userId, amount);
            return Map.of("code", 0, "userId", userId, "balance", bal);
        } catch (IllegalStateException e) {
            return Map.of("code", 2, "msg", e.getMessage());
        }
    }

    @GetMapping("/shop/{shopId}")
    public Map<String, Object> shop(@PathVariable long shopId) {
        Map<String, Object> row = shopMapper.selectById(shopId);
        if (row == null) {
            return Map.of("code", 404, "msg", "shop not found");
        }
        return Map.of("code", 0, "data", row);
    }
}
