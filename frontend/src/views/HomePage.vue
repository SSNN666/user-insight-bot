<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getProducts } from '../api'
import { useRouter } from 'vue-router'

const products = ref<any[]>([])
const router = useRouter()

onMounted(async () => {
  try {
    const d = await getProducts({ page_size: 8 })
    products.value = d.products || []
  } catch(e) { console.error(e) }
})
</script>

<template>
  <div class="home">
    <section class="hero">
      <h1>🛒 智能电商平台</h1>
      <p>AI 驱动的用户画像分析与个性化购物体验</p>
      <div class="hero-btns">
        <button class="btn-primary" @click="router.push('/products')">开始购物</button>
        <button class="btn-secondary" @click="router.push('/ai')">🤖 AI 助手</button>
      </div>
    </section>

    <section class="featured">
      <h2>推荐商品</h2>
      <div class="product-grid">
        <div v-for="p in products" :key="p.product_id" class="product-card" @click="router.push(`/product/${p.product_id}`)">
          <div class="prod-icon">📦</div>
          <h3>{{ p.product_name }}</h3>
          <span class="cat">{{ p.category }}</span>
          <p class="price">¥{{ p.price }}</p>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.hero { text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #409EFF, #67C23A); color: white; border-radius: 8px; margin-bottom: 40px; }
.hero h1 { font-size: 2.5rem; margin: 0 0 10px 0; }
.hero p { font-size: 1.2rem; margin-bottom: 20px; opacity: 0.9; }
.hero-btns { display: flex; gap: 12px; justify-content: center; }
.btn-primary { padding: 12px 32px; background: white; color: #409EFF; border: none; border-radius: 6px; font-size: 1rem; font-weight: bold; cursor: pointer; }
.btn-secondary { padding: 12px 32px; background: rgba(255,255,255,.2); color: white; border: 1px solid white; border-radius: 6px; font-size: 1rem; cursor: pointer; }
.product-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }
.product-card { border: 1px solid #eee; border-radius: 8px; padding: 20px; cursor: pointer; transition: box-shadow .2s; }
.product-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,.1); }
.prod-icon { font-size: 3rem; text-align: center; margin-bottom: 12px; }
.product-card h3 { margin: 0 0 8px 0; font-size: 1rem; }
.cat { color: #909399; font-size: 0.8rem; background: #f0f0f0; padding: 2px 8px; border-radius: 4px; }
.price { color: #E6A23C; font-size: 1.3rem; font-weight: bold; margin: 8px 0 0 0; }
</style>
