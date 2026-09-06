package com.seckill.mapper;

import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

/** 用户钱包：余额单位分；积分 points；支付扣减用 CAS（balance >= amount）。 */
@Mapper
public interface UserMapper {

    @Select("SELECT balance FROM tb_user WHERE user_id = #{userId}")
    Long selectBalance(@Param("userId") long userId);

    @Select("SELECT points FROM tb_user WHERE user_id = #{userId}")
    Long selectPoints(@Param("userId") long userId);

    @Insert("INSERT INTO tb_user(user_id, balance, points) VALUES(#{userId}, #{balance}, 0) " +
            "ON DUPLICATE KEY UPDATE user_id = user_id")
    int ensure(@Param("userId") long userId, @Param("balance") long balance);

    @Update("UPDATE tb_user SET balance = balance + #{amount} WHERE user_id = #{userId}")
    int credit(@Param("userId") long userId, @Param("amount") long amount);

    /** 支付扣减：余额不足或不存在则 0 行 */
    @Update("UPDATE tb_user SET balance = balance - #{amount} " +
            "WHERE user_id = #{userId} AND balance >= #{amount}")
    int deduct(@Param("userId") long userId, @Param("amount") long amount);

    /** 支付成功后异步加积分（Kafka 消费） */
    @Update("UPDATE tb_user SET points = points + #{points} WHERE user_id = #{userId}")
    int addPoints(@Param("userId") long userId, @Param("points") long points);
}
