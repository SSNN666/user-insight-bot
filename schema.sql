-- E-commerce database schema (reference DDL — the app uses in-memory mock data
-- for development and only connects to MySQL when DB_HOST is configured).
--
-- Usage: mysql -u root -p < schema.sql

CREATE DATABASE IF NOT EXISTS ecommerce DEFAULT CHARACTER SET utf8mb4;
USE ecommerce;

-- ── Users ──────────────────────────────────────────────────────
CREATE TABLE users (
    user_id   INT PRIMARY KEY AUTO_INCREMENT,
    username  VARCHAR(50) NOT NULL,
    reg_date  DATE NOT NULL,
    city      VARCHAR(50),
    age       TINYINT UNSIGNED,
    gender    ENUM('M','F','Other')
) ENGINE=InnoDB;

-- ── Products ───────────────────────────────────────────────────
CREATE TABLE products (
    product_id    INT PRIMARY KEY AUTO_INCREMENT,
    product_name  VARCHAR(100) NOT NULL,
    category      VARCHAR(50),
    price         DECIMAL(10,2)
) ENGINE=InnoDB;

-- ── Orders (partitioned by year) ───────────────────────────────
CREATE TABLE orders (
    order_id      INT PRIMARY KEY AUTO_INCREMENT,
    user_id       INT NOT NULL,
    product_id    INT NOT NULL,
    order_date    DATE NOT NULL,
    quantity      INT NOT NULL,
    total_amount  DECIMAL(10,2) NOT NULL,
    FOREIGN KEY (user_id)   REFERENCES users(user_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
) ENGINE=InnoDB
PARTITION BY RANGE (YEAR(order_date)) (
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION p2026 VALUES LESS THAN (2027),
    PARTITION p2027 VALUES LESS THAN (2028),
    PARTITION p2028 VALUES LESS THAN (2029),
    PARTITION p2029 VALUES LESS THAN (2030),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);

-- ── Sample data ────────────────────────────────────────────────
INSERT INTO users VALUES
(1,'alice','2024-01-15','Beijing',28,'F'),
(2,'bob',  '2024-06-20','Shanghai',35,'M'),
(3,'carol','2024-03-10','Guangzhou',22,'F'),
(4,'dave', '2025-01-05','Shenzhen',40,'M'),
(5,'eve',  '2025-02-01','Beijing',31,'F');

INSERT INTO products VALUES
(1,'Laptop','Electronics',5999.00),
(2,'Mouse','Electronics',99.00),
(3,'Book','Education',59.00),
(4,'Headphones','Electronics',299.00);

INSERT INTO orders VALUES
(1,1,1,'2025-02-20',1,5999.00),
(2,1,2,'2025-03-15',2,198.00),
(3,2,3,'2025-01-10',1,59.00),
(4,2,1,'2025-05-01',1,5999.00),
(5,3,4,'2026-01-12',1,299.00),
(6,1,3,'2026-02-01',2,118.00),
(7,4,1,'2026-04-20',1,5999.00),
(8,5,2,'2026-06-05',1,99.00),
(9,5,4,'2026-07-18',1,299.00);
