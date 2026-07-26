<script setup lang="ts">
import { useCartStore } from './stores/cart'
import { onMounted } from 'vue'

const cart = useCartStore()
onMounted(() => cart.fetch())
</script>

<template>
  <div>
    <header class="navbar">
      <router-link to="/" class="logo">🛒 智能电商</router-link>
      <nav class="nav-links">
        <router-link to="/products">商品</router-link>
        <router-link to="/cart">购物车 ({{ cart.itemCount }})</router-link>
        <router-link to="/orders">订单</router-link>
        <router-link to="/ai">🤖 AI</router-link>
        <span class="sep">|</span>
        <a href="http://localhost:7860" target="_blank">📊 管理后台</a>
        <a href="http://localhost:8000/docs" target="_blank">📘 API</a>
      </nav>
    </header>
    <main class="main-content">
      <router-view />
    </main>
  </div>
</template>

<style>
body { margin: 0; font-family: 'Helvetica Neue', Arial, sans-serif; }
.navbar {
  display: flex; align-items: center; justify-content: space-between;
  background: #409EFF; color: white; padding: 0 24px; height: 56px;
}
.logo { font-size: 1.3rem; font-weight: bold; color: white; text-decoration: none; }
.nav-links { display: flex; gap: 20px; }
.nav-links a { color: white; text-decoration: none; font-size: 0.95rem; }
.nav-links a:hover { text-decoration: underline; }
.main-content { max-width: 1200px; margin: 0 auto; padding: 24px; }
</style>
