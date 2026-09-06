package com.seckill.mapper;

import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.util.List;
import java.util.Map;

/**
 * 订单状态机：0 待支付 / 1 已支付 / 2 超时取消 / 3 用户取消。
 * 积分：已支付订单 points_status 0 待入账 / 1 已入账（Kafka 消费幂等）。
 */
@Mapper
public interface OrderMapper {

    @Insert("INSERT IGNORE INTO tb_voucher_order(order_no, voucher_id, user_id, status, points_status) " +
            "VALUES(#{orderNo}, #{voucherId}, #{userId}, 0, 0)")
    int insertUnpaid(@Param("orderNo") long orderNo, @Param("voucherId") long voucherId, @Param("userId") long userId);

    @Select("SELECT order_no FROM tb_voucher_order WHERE voucher_id = #{voucherId} AND user_id = #{userId}")
    Long selectOrderNo(@Param("voucherId") long voucherId, @Param("userId") long userId);

    @Select("SELECT status FROM tb_voucher_order WHERE voucher_id = #{voucherId} AND user_id = #{userId}")
    Integer selectStatus(@Param("voucherId") long voucherId, @Param("userId") long userId);

    @Select("SELECT order_no AS orderNo, voucher_id AS voucherId, user_id AS userId, amount, points_status AS pointsStatus " +
            "FROM tb_voucher_order WHERE voucher_id = #{voucherId} AND user_id = #{userId}")
    Map<String, Object> selectOne(@Param("voucherId") long voucherId, @Param("userId") long userId);

    /** 支付：仅待支付 → 已支付 */
    @Update("UPDATE tb_voucher_order SET status = 1, pay_time = NOW(), amount = #{amount}, shop_id = #{shopId}, " +
            "points_status = 0 " +
            "WHERE voucher_id = #{voucherId} AND user_id = #{userId} AND status = 0")
    int pay(@Param("voucherId") long voucherId, @Param("userId") long userId,
            @Param("amount") int amount, @Param("shopId") long shopId);

    /** 超时取消：待支付 → 2 */
    @Update("UPDATE tb_voucher_order SET status = 2, cancel_time = NOW() " +
            "WHERE voucher_id = #{voucherId} AND user_id = #{userId} AND status = 0")
    int cancelTimeout(@Param("voucherId") long voucherId, @Param("userId") long userId);

    /** 用户取消：待支付 → 3 */
    @Update("UPDATE tb_voucher_order SET status = 3, cancel_time = NOW() " +
            "WHERE voucher_id = #{voucherId} AND user_id = #{userId} AND status = 0")
    int cancelByUser(@Param("voucherId") long voucherId, @Param("userId") long userId);

    /** @deprecated 兼容旧名，等同超时取消 */
    @Update("UPDATE tb_voucher_order SET status = 2, cancel_time = NOW() " +
            "WHERE voucher_id = #{voucherId} AND user_id = #{userId} AND status = 0")
    int cancel(@Param("voucherId") long voucherId, @Param("userId") long userId);

    @Select("<script>SELECT voucher_id, user_id, status FROM tb_voucher_order WHERE " +
            "(voucher_id, user_id) IN " +
            "<foreach collection='keys' item='k' open='(' separator=',' close=')'>(#{k[0]},#{k[1]})</foreach>" +
            "</script>")
    List<Map<String, Object>> selectStatuses(@Param("keys") List<long[]> keys);

    @Update("<script>UPDATE tb_voucher_order SET status = 2, cancel_time = NOW() WHERE status = 0 AND " +
            "(voucher_id, user_id) IN " +
            "<foreach collection='keys' item='k' open='(' separator=',' close=')'>(#{k[0]},#{k[1]})</foreach>" +
            "</script>")
    int cancelBatch(@Param("keys") List<long[]> keys);

    /** Kafka 消费入账：CAS 标记积分已入账 */
    @Update("UPDATE tb_voucher_order SET points_status = 1 " +
            "WHERE order_no = #{orderNo} AND status = 1 AND points_status = 0")
    int markPointsCredited(@Param("orderNo") long orderNo);

    /** 已支付但积分未入账：中继重投 Kafka（订单表兼 Outbox，无独立消息表） */
    @Select("SELECT order_no AS orderNo, user_id AS userId, amount FROM tb_voucher_order " +
            "WHERE status = 1 AND points_status = 0 ORDER BY id LIMIT 500")
    List<Map<String, Object>> selectPointsPending();

    @Select("SELECT COUNT(*) FROM tb_voucher_order")
    long countAll();

    @Select("SELECT COUNT(*) FROM tb_voucher_order WHERE status = #{status}")
    long countByStatus(int status);

    @Select("SELECT status, COUNT(*) AS cnt FROM tb_voucher_order GROUP BY status")
    List<Map<String, Object>> countGroupByStatus();
}
