package com.seckill.config;

import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

/** 美食汇：启动时建表 + 造数（幂等）：商铺/用户（余额+积分）/券/订单。 */
@Component
public class SchemaInit {

    private static final int SHOP_COUNT = 100;
    private static final int USER_SEED = 10_000;
    private static final long USER_INIT_BALANCE = 10_000; // 分：100 元
    private static final int VOUCHER_PRICE = 100;          // 分：1 元/张

    private final JdbcTemplate jdbc;

    public SchemaInit(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    @jakarta.annotation.PostConstruct
    public void init() {
        Integer locked = jdbc.queryForObject("SELECT GET_LOCK('seckill_schema_init', 120)", Integer.class);
        if (locked == null || locked != 1) {
            throw new IllegalStateException("cannot acquire schema_init lock");
        }
        try {
            doInit();
        } finally {
            jdbc.queryForObject("SELECT RELEASE_LOCK('seckill_schema_init')", Integer.class);
        }
    }

    private void doInit() {
        jdbc.execute("CREATE TABLE IF NOT EXISTS tb_shop (" +
                "shop_id BIGINT PRIMARY KEY, name VARCHAR(64) NOT NULL, " +
                "balance BIGINT NOT NULL DEFAULT 0, " +
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, " +
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP)");

        jdbc.execute("CREATE TABLE IF NOT EXISTS tb_user (" +
                "user_id BIGINT PRIMARY KEY, " +
                "balance BIGINT NOT NULL DEFAULT 0, " +
                "points BIGINT NOT NULL DEFAULT 0, " +
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, " +
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP)");
        ensureUserPoints();

        jdbc.execute("CREATE TABLE IF NOT EXISTS tb_seckill_voucher (" +
                "voucher_id BIGINT PRIMARY KEY, available INT NOT NULL DEFAULT 0, " +
                "locked INT NOT NULL DEFAULT 0, sold INT NOT NULL DEFAULT 0, " +
                "version BIGINT NOT NULL DEFAULT 0, " +
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, " +
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP)");

        jdbc.execute("CREATE TABLE IF NOT EXISTS tb_voucher_detail (" +
                "voucher_id BIGINT PRIMARY KEY, shop_id BIGINT NOT NULL DEFAULT 1, " +
                "name VARCHAR(64) NOT NULL, description VARCHAR(255), " +
                "price INT NOT NULL DEFAULT 100, " +
                "KEY idx_shop (shop_id))");
        ensureDetailColumns();

        String[] cuisines = {"川菜", "粤菜", "淮扬菜", "鲁菜", "湘菜", "东北菜", "本帮菜", "日料", "西餐", "火锅"};
        jdbc.batchUpdate("INSERT INTO tb_voucher_detail(voucher_id, shop_id, name, description, price) VALUES(?,?,?,?,?) " +
                        "ON DUPLICATE KEY UPDATE shop_id = VALUES(shop_id), name = VALUES(name), " +
                        "description = VALUES(description), price = VALUES(price)",
                java.util.stream.IntStream.rangeClosed(1, 1000)
                        .mapToObj(i -> {
                            String c = cuisines[(i - 1) % cuisines.length];
                            long shopId = ((i - 1) % SHOP_COUNT) + 1L;
                            return new Object[]{(long) i, shopId,
                                    "美食汇·" + c + "满减券-" + i,
                                    "美食汇商铺" + shopId + "·" + c + "专场：满100减30，售价" + (VOUCHER_PRICE / 100.0) + "元",
                                    VOUCHER_PRICE};
                        })
                        .toList());

        jdbc.batchUpdate("INSERT INTO tb_shop(shop_id, name, balance) VALUES(?,?,0) " +
                        "ON DUPLICATE KEY UPDATE name = VALUES(name)",
                java.util.stream.IntStream.rangeClosed(1, SHOP_COUNT)
                        .mapToObj(i -> new Object[]{(long) i, "美食汇·商铺" + i})
                        .toList());

        jdbc.batchUpdate("INSERT INTO tb_user(user_id, balance, points) VALUES(?,?,0) " +
                        "ON DUPLICATE KEY UPDATE user_id = user_id",
                java.util.stream.IntStream.rangeClosed(1, USER_SEED)
                        .mapToObj(i -> new Object[]{(long) i, USER_INIT_BALANCE})
                        .toList());

        jdbc.execute("CREATE TABLE IF NOT EXISTS tb_voucher_order (" +
                "id BIGINT AUTO_INCREMENT PRIMARY KEY, order_no BIGINT NOT NULL UNIQUE, " +
                "voucher_id BIGINT NOT NULL, user_id BIGINT NOT NULL, " +
                "status TINYINT NOT NULL DEFAULT 0, " +
                "points_status TINYINT NOT NULL DEFAULT 0, " +
                "amount INT NULL, shop_id BIGINT NULL, " +
                "pay_time DATETIME NULL, cancel_time DATETIME NULL, " +
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, " +
                "UNIQUE KEY uk_voucher_user (voucher_id, user_id), " +
                "KEY idx_points_pending (status, points_status))");
        ensureOrderColumns();

        try {
            jdbc.execute("DROP TABLE IF EXISTS tb_local_message");
        } catch (Exception ignored) {
        }

        jdbc.batchUpdate("INSERT INTO tb_seckill_voucher(voucher_id, available) " +
                        "VALUES(?, 1000) ON DUPLICATE KEY UPDATE available = 1000, locked = 0, sold = 0, version = version + 1",
                java.util.stream.IntStream.rangeClosed(1, 1000)
                        .mapToObj(i -> new Object[]{(long) i})
                        .toList());
        jdbc.execute("TRUNCATE tb_voucher_order");
        jdbc.update("UPDATE tb_shop SET balance = 0");
        jdbc.update("UPDATE tb_user SET balance = ?, points = 0 WHERE user_id <= ?", USER_INIT_BALANCE, USER_SEED);
    }

    private void ensureUserPoints() {
        try {
            jdbc.execute("ALTER TABLE tb_user ADD COLUMN points BIGINT NOT NULL DEFAULT 0");
        } catch (Exception ignored) {
        }
    }

    private void ensureDetailColumns() {
        try {
            jdbc.execute("ALTER TABLE tb_voucher_detail ADD COLUMN shop_id BIGINT NOT NULL DEFAULT 1");
        } catch (Exception ignored) {
        }
        try {
            jdbc.execute("ALTER TABLE tb_voucher_detail ADD COLUMN price INT NOT NULL DEFAULT 100");
        } catch (Exception ignored) {
        }
    }

    private void ensureOrderColumns() {
        try {
            jdbc.execute("ALTER TABLE tb_voucher_order ADD COLUMN amount INT NULL");
        } catch (Exception ignored) {
        }
        try {
            jdbc.execute("ALTER TABLE tb_voucher_order ADD COLUMN shop_id BIGINT NULL");
        } catch (Exception ignored) {
        }
        try {
            jdbc.execute("ALTER TABLE tb_voucher_order ADD COLUMN points_status TINYINT NOT NULL DEFAULT 0");
        } catch (Exception ignored) {
        }
    }
}
