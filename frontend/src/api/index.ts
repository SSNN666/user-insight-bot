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

/**
 * SSE 流式问答:逐步展示 Agent 执行过程。
 * 事件:meta / node_start / node_end_detail / tool_call / tool_result /
 *       delta / fact_check / answer / done / error
 * axios 不支持 SSE 帧解析 → 原生 fetch + 手写帧拆分(零新依赖)。
 */
export function askAgentStream(
  question: string,
  sessionId: string,
  onEvent: (ev: any) => void,
  signal?: AbortSignal,
): Promise<void> {
  return fetch('/ask/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId }),
    signal,
  }).then(async res => {
    if (!res.ok || !res.body) {
      onEvent({ event: 'error', message: `HTTP ${res.status}` })
      return
    }
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      let idx: number
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const frame = buf.slice(0, idx)
        buf = buf.slice(idx + 2)
        for (const line of frame.split('\n')) {
          if (line.startsWith('data: ')) {
            try { onEvent(JSON.parse(line.slice(6))) } catch { /* 忽略坏帧 */ }
          }
        }
      }
    }
  })
}

export default api
