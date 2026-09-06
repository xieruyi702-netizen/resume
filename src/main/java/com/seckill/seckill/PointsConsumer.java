package com.seckill.seckill;

import com.seckill.mapper.OrderMapper;
import com.seckill.mapper.UserMapper;
import jakarta.annotation.PreDestroy;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

import java.time.Duration;
import java.util.List;
import java.util.Properties;

/**
 * 支付积分消费者：消息 orderNo:userId:points。
 * 同一事务内 CAS 标记 + 加积分；重复消费 mark 失败则跳过。失败重试耗尽进 DLT。
 */
@Component
public class PointsConsumer {

    private final KafkaConsumer<Long, String> consumer;
    private final OrderMapper orderMapper;
    private final UserMapper userMapper;
    private final TransactionTemplate tx;
    private final Thread worker;
    private volatile boolean running = true;
    private final KafkaProducer<Long, String> deadProducer;
    private final String dltTopic;
    private static final int MAX_RETRY = 3;

    public PointsConsumer(OrderMapper orderMapper, UserMapper userMapper,
                          PlatformTransactionManager ptm,
                          @Value("${seckill.kafka.bootstrap}") String bootstrap,
                          @Value("${seckill.kafka.topic}") String topic) {
        this.orderMapper = orderMapper;
        this.userMapper = userMapper;
        this.tx = new TransactionTemplate(ptm);
        Properties p = new Properties();
        p.put("bootstrap.servers", bootstrap);
        p.put("group.id", "seckill-points");
        p.put("key.deserializer", "org.apache.kafka.common.serialization.LongDeserializer");
        p.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
        p.put("enable.auto.commit", "false");
        p.put("auto.offset.reset", "earliest");
        this.consumer = new KafkaConsumer<>(p);
        this.consumer.subscribe(List.of(topic));

        Properties dp = new Properties();
        dp.put("bootstrap.servers", bootstrap);
        dp.put("key.serializer", "org.apache.kafka.common.serialization.LongSerializer");
        dp.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        dp.put("acks", "all");
        this.deadProducer = new KafkaProducer<>(dp);
        this.dltTopic = topic + "-dlt";

        this.worker = new Thread(this::run, "points-consumer");
        this.worker.setDaemon(true);
        this.worker.start();
    }

    private void run() {
        while (consumer.assignment().isEmpty()) {
            consumer.poll(Duration.ofMillis(100));
        }
        while (running) {
            ConsumerRecords<Long, String> records = consumer.poll(Duration.ofMillis(200));
            for (ConsumerRecord<Long, String> r : records) {
                handleWithDlt(r.value());
            }
            if (!records.isEmpty()) consumer.commitSync();
        }
        consumer.close();
        deadProducer.close();
    }

    void handleWithDlt(String msg) {
        for (int attempt = 1; attempt <= MAX_RETRY; attempt++) {
            try {
                handle(msg);
                return;
            } catch (Exception e) {
                if (attempt == MAX_RETRY) {
                    try {
                        deadProducer.send(new org.apache.kafka.clients.producer.ProducerRecord<>(
                                dltTopic, 0L, msg)).get(5, java.util.concurrent.TimeUnit.SECONDS);
                        System.err.println("[DLT] points retry exhausted → " + dltTopic + ": " + msg + " / " + e);
                    } catch (Exception sendFail) {
                        System.err.println("[DLT] points DLT send failed: " + msg + " / " + sendFail);
                    }
                } else {
                    try { Thread.sleep(50L * attempt); } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        return;
                    }
                }
            }
        }
    }

    void handle(String msg) {
        String[] parts = msg.split(":");
        if (parts.length < 3) {
            throw new IllegalArgumentException("bad points msg: " + msg);
        }
        long orderNo = Long.parseLong(parts[0]);
        long userId = Long.parseLong(parts[1]);
        long points = Long.parseLong(parts[2]);
        if (points <= 0) return;
        tx.executeWithoutResult(s -> {
            if (orderMapper.markPointsCredited(orderNo) == 0) {
                return;
            }
            userMapper.addPoints(userId, points);
        });
    }

    @PreDestroy
    public void shutdown() {
        running = false;
        worker.interrupt();
    }
}
