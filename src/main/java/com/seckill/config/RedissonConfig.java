package com.seckill.config;

import org.redisson.Redisson;
import org.redisson.api.RedissonClient;
import org.redisson.config.Config;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/** Redisson：用户余额缓存 + 钱包分布式锁（与现有 Jedis 秒杀链路并存）。 */
@Configuration
public class RedissonConfig {

    @Bean(destroyMethod = "shutdown")
    public RedissonClient redissonClient(@Value("${seckill.redis.master}") String master) {
        String[] hp = master.split(":");
        Config config = new Config();
        config.useSingleServer()
                .setAddress("redis://" + hp[0] + ":" + hp[1])
                .setConnectionMinimumIdleSize(4)
                .setConnectionPoolSize(32);
        return Redisson.create(config);
    }
}
