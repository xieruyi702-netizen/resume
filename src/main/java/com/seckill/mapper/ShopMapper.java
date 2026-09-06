package com.seckill.mapper;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.util.Map;

/** 商铺账户：支付成功后按券售价入账。 */
@Mapper
public interface ShopMapper {

    @Select("SELECT shop_id AS shopId, name, balance FROM tb_shop WHERE shop_id = #{shopId}")
    Map<String, Object> selectById(@Param("shopId") long shopId);

    @Update("UPDATE tb_shop SET balance = balance + #{amount} WHERE shop_id = #{shopId}")
    int credit(@Param("shopId") long shopId, @Param("amount") long amount);
}
