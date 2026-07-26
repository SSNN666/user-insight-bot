<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getProducts, getCategories, addToCart } from '../api'
import { useRouter } from 'vue-router'

const products = ref<any[]>([])
const categories = ref<string[]>([])
const selectedCategory = ref('')
const searchQuery = ref('')
const loading = ref(true)
const cartCount = ref(0)
const router = useRouter()

async function load() {
  loading.value = true
  try {
    const [prodData, catData] = await Promise.all([
      getProducts({ page_size: 50 }),
      getCategories(),
    ])
    products.value = prodData.products || []
    categories.value = catData.categories || []
  } catch(e) { console.error(e) }
  finally { loading.value = false }
}

onMounted(load)

async function filter() {
  loading.value = true
  try {
    const data = await getProducts({
      category: selectedCategory.value || undefined,
      q: searchQuery.value || undefined,
      page_size: 50,
    })
    products.value = data.products || []
  } catch(e) { console.error(e) }
  finally { loading.value = false }
}

async function add(pid: number) {
  try {
    await addToCart(pid)
    cartCount.value++
    alert('已加入购物车')
  } catch(e) { console.error(e) }
}
</script>

<template>
  <div>
    <h2>商品列表</h2>
    <div class="filters">
      <select v-model="selectedCategory" @change="filter">
        <option value="">全部分类</option>
        <option v-for="c in categories" :key="c" :value="c">{{ c }}</option>
      </select>
      <input v-model="searchQuery" placeholder="搜索..." @keyup.enter="filter" />
      <button @click="filter">搜索</button>
    </div>

    <div v-if="loading" class="loading">加载中...</div>

    <div v-else class="product-grid">
      <div v-for="p in products" :key="p.product_id" class="product-card" @click="router.push(`/product/${p.product_id}`)">
        <div class="product-icon">📦</div>
        <h3>{{ p.product_name }}</h3>
        <span class="cat">{{ p.category }}</span>
        <p class="price">¥{{ p.price }}</p>
        <button class="btn-cart" @click.stop="add(p.product_id)">加入购物车</button>
      </div>
    </div>

    <div v-if="!loading && !products.length" style="text-align:center;padding:40px;color:#999">暂无商品</div>
  </div>
</template>

<style scoped>
.filters { display: flex; gap: 12px; margin-bottom: 24px; }
.filters select, .filters input { padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
.filters button { padding: 8px 20px; background: #409EFF; color: white; border: none; border-radius: 4px; cursor: pointer; }
.product-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }
.product-card { border: 1px solid #eee; border-radius: 8px; padding: 20px; cursor: pointer; transition: box-shadow .2s; }
.product-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,.1); }
.product-icon { font-size: 3rem; text-align: center; margin-bottom: 12px; }
.product-card h3 { margin: 0 0 8px 0; font-size: 1rem; }
.cat { color: #909399; font-size: 0.8rem; background: #f0f0f0; padding: 2px 8px; border-radius: 4px; }
.price { color: #E6A23C; font-size: 1.3rem; font-weight: bold; margin: 8px 0; }
.btn-cart { width: 100%; padding: 8px; background: #409EFF; color: white; border: none; border-radius: 4px; cursor: pointer; }
.loading { text-align: center; padding: 40px; color: #999; }
</style>
