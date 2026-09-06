package com.seckill.seckill;

import com.seckill.mapper.OrderMapper;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/**
 * 积分投递中继：扫描「已支付且积分未入账」订单重投 Kafka。
 * 用订单表自身充当 Outbox，无需独立消息表。
 */
@Component
public class PointsRelay {

    private final OrderMapper orderMapper;
    private final PayService payService;

    public PointsRelay(OrderMapper orderMapper, PayService payService) {
        this.orderMapper = orderMapper;
        this.payService = payService;
    }

    @Scheduled(fixedDelay = 1000)
    public void relay() {
        for (var row : orderMapper.selectPointsPending()) {
            long orderNo = ((Number) row.get("orderNo")).longValue();
            long userId = ((Number) row.get("userId")).longValue();
            Object amountObj = row.get("amount");
            int points = amountObj == null ? 0 : ((Number) amountObj).intValue();
            payService.publishPoints(orderNo, userId, points);
        }
    }
}
