import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { getCart, addToCart, removeFromCart, clearCart } from '../api'

export const useCartStore = defineStore('cart', () => {
  const items = ref<any[]>([])
  const loading = ref(false)

  const itemCount = computed(() => items.value.reduce((s, i) => s + i.quantity, 0))
  const total = computed(() => items.value.reduce((s, i) => s + i.price * i.quantity, 0))

  async function fetch() {
    loading.value = true
    try {
      const data = await getCart()
      items.value = data.items || []
    } finally {
      loading.value = false
    }
  }

  async function add(pid: number, qty = 1) {
    const data = await addToCart(pid, qty)
    items.value = data.items || []
  }

  async function remove(pid: number) {
    const data = await removeFromCart(pid)
    items.value = data.items || []
  }

  async function clear() {
    await clearCart()
    items.value = []
  }

  return { items, loading, itemCount, total, fetch, add, remove, clear }
})
