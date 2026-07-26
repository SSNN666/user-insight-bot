<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { askAgent } from '../api'

const messages = ref<{ role: string; content: string }[]>([])
const input = ref('')
const loading = ref(false)

// Welcome message
onMounted(() => {
  messages.value.push({
    role: 'assistant',
    content: '你好！我是购物助手 🛍️\n\n我可以帮你：\n- 🔍 **搜商品**：输入商品名帮你找\n- 📂 **逛品类**：看看每个品类有什么\n- 💰 **比价格**：对比同类商品价格\n\n直接告诉我你想找什么吧！',
  })
})

async function send() {
  if (!input.value.trim()) return
  const q = input.value; input.value = ''
  messages.value.push({ role: 'user', content: q })
  loading.value = true
  try {
    const d = await askAgent(q)
    messages.value.push({ role: 'assistant', content: d.reply })
  } catch {
    messages.value.push({ role: 'assistant', content: '抱歉，出了点问题，请稍后再试。' })
  } finally { loading.value = false }
}
</script>

<template>
  <div>
    <h2>🤖 AI 购物助手</h2>
    <div class="chat-window">
      <div v-for="(m, i) in messages" :key="i" :class="['msg', m.role]">
        <strong>{{ m.role === 'user' ? '你' : 'AI' }}:</strong>
        <div v-text="m.content" style="white-space:pre-wrap" />
      </div>
      <div v-if="loading" class="msg assistant"><em>正在帮你找...</em></div>
    </div>
    <div class="chat-input">
      <div class="quick-actions">
        <button class="quick-btn" @click="input='电子产品'; send()" :disabled="loading">📱 电子产品</button>
        <button class="quick-btn" @click="input='时尚服饰'; send()" :disabled="loading">👗 时尚服饰</button>
        <button class="quick-btn" @click="input='有什么推荐的？'; send()" :disabled="loading">🎯 推荐给我</button>
      </div>
      <div class="input-row">
        <input v-model="input" @keyup.enter="send" placeholder="输入商品名或购物需求..." :disabled="loading" />
        <button @click="send" :disabled="loading">发送</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat-window { height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; background: #fafafa; }
.msg { margin-bottom: 12px; padding: 8px 12px; border-radius: 8px; max-width: 80%; }
.msg.user { background: #409EFF; color: white; margin-left: auto; }
.msg.assistant { background: #e8e8e8; }
.chat-input { display: flex; flex-direction: column; gap: 10px; }
.input-row { display: flex; gap: 12px; }
.input-row input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
.input-row button { padding: 10px 24px; background: #409EFF; color: white; border: none; border-radius: 4px; cursor: pointer; }
.quick-actions { display: flex; gap: 8px; }
.quick-btn { padding: 6px 14px; background: #f0f0f0; border: 1px solid #ddd; border-radius: 20px; cursor: pointer; font-size: 13px; transition: background .2s; }
.quick-btn:hover { background: #e0e0e0; }
</style>
