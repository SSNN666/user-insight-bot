<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { getProduct, addToCart } from '../api'

const route = useRoute()
const router = useRouter()
const product = ref<any>(null)
const qty = ref(1)

onMounted(async () => {
  try { product.value = await getProduct(Number(route.params.id)) } catch(e) { console.error(e) }
})

async function add() {
  try {
    await addToCart(product.value.product_id, qty.value)
    alert(`已添加 ${qty.value} 件到购物车`)
  } catch(e: any) { alert('添加失败') }
}
</script>

<template>
  <div v-if="product">
    <button class="back" @click="router.back()">← 返回</button>
    <div class="detail-row">
      <div class="prod-image">📦</div>
      <div class="prod-info">
        <h1>{{ product.product_name }}</h1>
        <span class="cat">{{ product.category }}</span>
        <p class="price">¥{{ product.price }}</p>
        <div class="actions">
          <input type="number" v-model="qty" :min="1" :max="99" class="qty" />
          <button class="btn-add" @click="add">加入购物车</button>
        </div>
      </div>
    </div>
  </div>
  <p v-else style="text-align:center;padding:60px;color:#999">加载中...</p>
</template>

<style scoped>
.back { background: none; border: none; color: #409EFF; cursor: pointer; font-size: 1rem; margin-bottom: 16px; }
.detail-row { display: flex; gap: 40px; }
.prod-image { font-size: 10rem; text-align: center; padding: 60px; background: #f5f7fa; border-radius: 12px; flex: 1; }
.prod-info { flex: 1; }
.prod-info h1 { font-size: 1.8rem; margin: 0 0 12px 0; }
.cat { display: inline-block; background: #f0f0f0; padding: 4px 12px; border-radius: 4px; color: #666; }
.price { font-size: 2rem; color: #E6A23C; font-weight: bold; margin: 20px 0; }
.actions { display: flex; gap: 16px; align-items: center; }
.qty { width: 60px; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 1rem; text-align: center; }
.btn-add { padding: 12px 32px; background: #409EFF; color: white; border: none; border-radius: 6px; font-size: 1.1rem; cursor: pointer; }
</style>
