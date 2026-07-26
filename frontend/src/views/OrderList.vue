<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getOrders, payOrder } from '../api'

const orders = ref<any[]>([])

onMounted(async () => {
  try { const d = await getOrders({ page_size: 50 }); orders.value = d.orders || [] } catch(e) { console.error(e) }
})

async function pay(oid: number) {
  try { await payOrder(oid); const o = orders.value.find(o => o.order_id === oid); if (o) o.status = '已支付' } catch(e) { console.error(e) }
}
</script>

<template>
  <div>
    <h2>📦 我的订单</h2>
    <table v-if="orders.length" class="order-table">
      <thead><tr><th>订单号</th><th>商品</th><th>金额</th><th>状态</th><th>操作</th></tr></thead>
      <tbody>
        <tr v-for="o in orders" :key="o.order_id">
          <td>#{{ o.order_id }}</td>
          <td>{{ o.items?.map((i: any) => i.product_name).join(', ') }}</td>
          <td>¥{{ o.total_amount }}</td>
          <td><span :class="o.status === '已支付' ? 'paid' : 'pending'">{{ o.status }}</span></td>
          <td>
            <button v-if="o.status !== '已支付'" class="btn-pay" @click="pay(o.order_id)">支付</button>
            <span v-else>✅</span>
          </td>
        </tr>
      </tbody>
    </table>
    <p v-else style="color:#999;padding:40px;text-align:center">暂无订单</p>
  </div>
</template>

<style scoped>
.order-table { width: 100%; border-collapse: collapse; }
.order-table th, .order-table td { padding: 12px; border-bottom: 1px solid #eee; text-align: left; }
.paid { color: #67C23A; font-weight: bold; }
.pending { color: #E6A23C; }
.btn-pay { padding: 4px 16px; background: #67C23A; color: white; border: none; border-radius: 4px; cursor: pointer; }
</style>
