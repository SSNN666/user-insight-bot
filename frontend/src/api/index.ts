/** API client — all calls proxy to FastAPI backend. */
import axios from 'axios'

const api = axios.create({
  baseURL: '',  // relative → goes through Vite proxy → localhost:8000
  timeout: 30000,
})

// ── Products ──
export const getProducts = (params?: Record<string, any>) =>
  api.get('/api/products', { params }).then(r => r.data)

export const getProduct = (id: number) =>
  api.get(`/api/products/${id}`).then(r => r.data)

export const getCategories = () =>
  api.get('/api/products/categories').then(r => r.data)

// ── Cart ──
export const getCart = (userId = 1) =>
  api.get('/api/cart', { params: { user_id: userId } }).then(r => r.data)

export const addToCart = (productId: number, quantity = 1, userId = 1) =>
  api.post('/api/cart/items', { product_id: productId, quantity }, { params: { user_id: userId } }).then(r => r.data)

export const removeFromCart = (productId: number, userId = 1) =>
  api.delete(`/api/cart/items/${productId}`, { params: { user_id: userId } }).then(r => r.data)

export const clearCart = (userId = 1) =>
  api.delete('/api/cart', { params: { user_id: userId } }).then(r => r.data)

// ── Orders ──
export const createOrder = (userId = 1) =>
  api.post('/api/orders', null, { params: { user_id: userId } }).then(r => r.data)

export const getOrders = (params?: Record<string, any>) =>
  api.get('/api/orders', { params }).then(r => r.data)

export const payOrder = (orderId: number) =>
  api.post(`/api/orders/${orderId}/pay`).then(r => r.data)

// ── User ──
export const getUserProfile = (userId = 1) =>
  api.get('/api/users/me', { params: { user_id: userId } }).then(r => r.data)

// ── AI Chat ──
export const askAgent = (question: string, sessionId = 'vue-chat') =>
  api.post('/ask', { question, session_id: sessionId }).then(r => r.data)

export default api
