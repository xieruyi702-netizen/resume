package com.seckill.mapper;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.util.Map;

@Mapper
public interface VoucherDetailMapper {

    @Select("SELECT CONCAT(name, ' | shop=', shop_id, ' | price=', price, '分 | ', description) " +
            "FROM tb_voucher_detail WHERE voucher_id = #{voucherId}")
    String selectDetail(long voucherId);

    /** 支付用：shop_id + price（分） */
    @Select("SELECT shop_id AS shopId, price FROM tb_voucher_detail WHERE voucher_id = #{voucherId}")
    Map<String, Object> selectPayInfo(@Param("voucherId") long voucherId);

    @Update("UPDATE tb_voucher_detail SET name = #{name}, description = #{description} " +
            "WHERE voucher_id = #{voucherId}")
    int updateDetail(@Param("voucherId") long voucherId,
                     @Param("name") String name,
                     @Param("description") String description);
}
