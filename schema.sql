-- 创建数据库
CREATE DATABASE IF NOT EXISTS ecommerce DEFAULT CHARACTER SET utf8mb4;
USE ecommerce;

-- 用户表
CREATE TABLE users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL,
    reg_date DATE NOT NULL,
    city VARCHAR(50),
    age TINYINT,
    gender ENUM('M','F','Other')
) ENGINE=InnoDB;

-- 商品表
CREATE TABLE products (
    product_id INT PRIMARY KEY AUTO_INCREMENT,
    product_name VARCHAR(100) NOT NULL,
    category VARCHAR(50),
    price DECIMAL(10,2)
) ENGINE=InnoDB;

-- 订单表（按年份范围分区）
CREATE TABLE orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    product_id INT NOT NULL,
    order_date DATE NOT NULL,
    quantity INT NOT NULL,
    total_amount DECIMAL(10,2) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
) ENGINE=InnoDB
PARTITION BY RANGE (YEAR(order_date)) (
    PARTITION p2020 VALUES LESS THAN (2021),
    PARTITION p2021 VALUES LESS THAN (2022),
    PARTITION p2022 VALUES LESS THAN (2023),
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);

-- 插入示例数据（模拟）
INSERT INTO users VALUES
(1,'alice','2022-01-15','Beijing',28,'F'),
(2,'bob','2021-06-20','Shanghai',35,'M'),
(3,'carol','2023-03-10','Guangzhou',22,'F'),
(4,'dave','2020-11-05','Shenzhen',40,'M'),
(5,'eve','2022-09-01','Beijing',31,'F');

INSERT INTO products VALUES
(1,'Laptop','Electronics',5999.00),
(2,'Mouse','Electronics',99.00),
(3,'Book','Education',59.00),
(4,'Headphones','Electronics',299.00);

INSERT INTO orders VALUES
(1,1,1,'2022-02-20',1,5999.00),
(2,1,2,'2022-03-15',2,198.00),
(3,2,3,'2021-07-10',1,59.00),
(4,2,1,'2022-09-01',1,5999.00),
(5,3,4,'2023-05-12',1,299.00),
(6,1,3,'2022-12-01',2,118.00),
(7,4,1,'2021-11-20',1,5999.00),
(8,5,2,'2023-01-05',1,99.00),
(9,5,4,'2023-02-18',1,299.00);