import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'home',
      component: () => import('../views/HomePage.vue'),
    },
    {
      path: '/products',
      name: 'products',
      component: () => import('../views/ProductList.vue'),
    },
    {
      path: '/product/:id',
      name: 'product-detail',
      component: () => import('../views/ProductDetail.vue'),
    },
    {
      path: '/cart',
      name: 'cart',
      component: () => import('../views/CartPage.vue'),
    },
    {
      path: '/orders',
      name: 'orders',
      component: () => import('../views/OrderList.vue'),
    },
    {
      path: '/ai',
      name: 'ai-assistant',
      component: () => import('../views/AiChat.vue'),
    },
  ],
})

export default router
