<script setup lang="ts">
import { onMounted } from 'vue'
import { useCartStore } from '../stores/cart'
import { createOrder } from '../api'
import { useRouter } from 'vue-router'

const cart = useCartStore()
const router = useRouter()
onMounted(() => cart.fetch())

async function checkout() {
  try {
    const order = await createOrder()
    alert(`订单 #${order.order_id} 已创建，金额 ¥${order.total_amount}`)
    router.push('/orders')
  } catch(e: any) { alert('下单失败: ' + (e.response?.data?.detail || e.message)) }
}
</script>

<template>
  <div>
    <h2>🛒 购物车</h2>
    <table v-if="cart.items.length" class="cart-table">
      <thead><tr><th>商品</th><th>单价</th><th>数量</th><th>小计</th><th>操作</th></tr></thead>
      <tbody>
        <tr v-for="i in cart.items" :key="i.product_id">
          <td>{{ i.product_name }}</td>
          <td>¥{{ i.price }}</td>
          <td>{{ i.quantity }}</td>
          <td>¥{{ (i.price * i.quantity).toFixed(2) }}</td>
          <td><button class="btn-del" @click="cart.remove(i.product_id)">删除</button></td>
        </tr>
      </tbody>
    </table>
    <p v-else style="color:#999;padding:40px;text-align:center">购物车为空</p>
    <div v-if="cart.items.length" class="cart-footer">
      <span>合计: <strong>¥{{ cart.total.toFixed(2) }}</strong> ({{ cart.itemCount }} 件)</span>
      <button @click="cart.clear()">清空</button>
      <button class="btn-checkout" @click="checkout">下单结算</button>
    </div>
  </div>
</template>

<style scoped>
.cart-table { width: 100%; border-collapse: collapse; }
.cart-table th, .cart-table td { padding: 12px; border-bottom: 1px solid #eee; text-align: left; }
.btn-del { padding: 4px 12px; background: #f56c6c; color: white; border: none; border-radius: 4px; cursor: pointer; }
.cart-footer { margin-top: 24px; display: flex; align-items: center; gap: 16px; justify-content: flex-end; }
.cart-footer strong { color: #E6A23C; font-size: 1.3rem; }
.btn-checkout { padding: 12px 32px; background: #67C23A; color: white; border: none; border-radius: 6px; font-size: 1.1rem; cursor: pointer; }
button { padding: 6px 16px; border: 1px solid #ddd; border-radius: 4px; cursor: pointer; background: white; }
</style>
