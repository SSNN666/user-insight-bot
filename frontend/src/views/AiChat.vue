<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { askAgentStream, addToCart } from '../api'
import MiniChart from '../components/MiniChart.vue'

interface Product { product_id: number; product_name: string; category: string; price: number }
interface ChartSpec { type: 'bar' | 'line'; title: string; categories: string[]; series: { name: string; data: (number | null)[] }[] }
interface ChatMsg { role: string; content: string; products?: Product[]; charts?: ChartSpec[] }
interface Step { node: string; label: string; detail?: string; status: 'running' | 'done' }

const messages = ref<ChatMsg[]>([])
const steps = ref<Step[]>([])
const input = ref('')
const loading = ref(false)
const cartFeedback = ref('')

async function addProductToCart(p: Product) {
  try {
    await addToCart(p.product_id, 1)
    cartFeedback.value = `✅ 「${p.product_name}」已加入购物车`
    setTimeout(() => { cartFeedback.value = '' }, 2500)
  } catch {
    cartFeedback.value = '❌ 加购失败,请重试'
  }
}

const NODE_LABELS: Record<string, string> = {
  preprocess: '🔍 意图识别',
  llm_decide: '🤖 模型决策',
  tools: '🔧 工具调用',
  reflect: '🪞 数据校验',
  respond: '💬 生成回答',
  fact_check: '✅ 事实核查',
}

let abortCtrl: AbortController | null = null

// Welcome message
onMounted(() => {
  messages.value.push({
    role: 'assistant',
    content: '你好！我是购物助手 🛍️\n\n我可以帮你：\n- 🔍 **搜商品**：输入商品名帮你找\n- 📂 **逛品类**：看看每个品类有什么\n- 💰 **比价格**：对比同类商品价格\n\n直接告诉我你想找什么吧！',
  })
})

onUnmounted(() => { abortCtrl?.abort() })

function currentStep(node: string): Step | undefined {
  return steps.value.find(s => s.node === node && s.status === 'running')
}

function handleEvent(ev: any, assistantMsg: ChatMsg) {
  switch (ev.event) {
    case 'node_start': {
      const label = NODE_LABELS[ev.node] || ev.node
      // 同节点重复进入(reflect 循环)时刷新为新的 running 步
      const running = currentStep(ev.node)
      if (running) running.status = 'done'
      steps.value.push({ node: ev.node, label, status: 'running' })
      break
    }
    case 'node_end_detail': {
      const step = currentStep(ev.node)
      if (step) {
        if (ev.node === 'preprocess') step.detail = `意图: ${ev.intent} · 子任务 ${ev.sub_queries}`
        else if (ev.node === 'reflect') step.detail = ev.is_complete ? '数据完整' : '数据不完整，准备补查'
      }
      break
    }
    case 'tool_call': {
      const step = steps.value.find(s => s.node === 'tools' && s.status === 'running')
      if (step) {
        step.detail = step.detail || ''
        step.detail += `调用 ${ev.name}(${JSON.stringify(ev.args)})`
      }
      break
    }
    case 'tool_result': {
      const step = steps.value.find(s => s.node === 'tools' && s.status === 'running')
      if (step) step.detail = `${step.detail || ''} → ${ev.status} (${ev.elapsed_ms}ms)`
      break
    }
    case 'delta':
      assistantMsg.content += ev.content
      break
    case 'fact_check': {
      const step = steps.value.find(s => s.node === 'fact_check' && s.status === 'running')
      if (step) step.detail = ev.passed ? '通过' : `⚠️ ${ev.violations} 个违规`
      break
    }
    case 'answer':
      assistantMsg.content = ev.content   // 权威全文覆盖增量区
      break
    case 'error':
      assistantMsg.content = assistantMsg.content || `抱歉，出了点问题：${ev.message || ''}`
      break
    case 'done':
      if (ev.tools_called?.length) {
        const step = steps.value.find(s => s.node === 'tools' && s.status === 'running')
        if (step) step.detail = `共调用 ${ev.tools_called.length} 个工具`
      }
      if (Array.isArray(ev.products) && ev.products.length) {
        assistantMsg.products = ev.products.slice(0, 5)
      }
      if (Array.isArray(ev.charts) && ev.charts.length) {
        assistantMsg.charts = ev.charts.slice(0, 2)
      }
      break
  }
}

async function send() {
  if (!input.value.trim() || loading.value) return
  const q = input.value; input.value = ''
  messages.value.push({ role: 'user', content: q })
  const assistantMsg: ChatMsg = { role: 'assistant', content: '' }
  messages.value.push(assistantMsg)
  steps.value = []
  loading.value = true
  abortCtrl = new AbortController()
  try {
    await askAgentStream(q, 'vue-chat', ev => handleEvent(ev, assistantMsg), abortCtrl.signal)
  } catch {
    if (!assistantMsg.content) assistantMsg.content = '抱歉，出了点问题，请稍后再试。'
  } finally {
    // 所有 running 步标记完成
    steps.value.forEach(s => { s.status = 'done' })
    loading.value = false
    abortCtrl = null
  }
}
</script>

<template>
  <div>
    <h2>🤖 AI 购物助手</h2>
    <div class="chat-window">
      <div v-for="(m, i) in messages" :key="i" :class="['msg', m.role]">
        <strong>{{ m.role === 'user' ? '你' : 'AI' }}:</strong>
        <div v-text="m.content" style="white-space:pre-wrap" />
        <!-- 分析图表:分群统计/趋势,零依赖 SVG 渲染 -->
        <MiniChart v-for="(c, ci) in m.charts" :key="'c' + ci" :chart="c" />
        <!-- 商品卡片:搜索/推荐结果一键加购 -->
        <div v-if="m.products?.length" class="product-cards">
          <div v-for="p in m.products" :key="p.product_id" class="product-card">
            <div class="p-name">{{ p.product_name }}</div>
            <div class="p-meta">{{ p.category }} · ¥{{ p.price }}</div>
            <button class="p-btn" @click="addProductToCart(p)">🛒 加购</button>
          </div>
        </div>
      </div>
      <div v-if="loading" class="msg assistant"><em>正在帮你找...</em></div>
    </div>

    <div v-if="cartFeedback" class="cart-feedback">{{ cartFeedback }}</div>

    <!-- Agent 执行步骤条 -->
    <div v-if="steps.length" class="steps">
      <span v-for="(s, i) in steps" :key="i" :class="['step', s.status]">
        <span v-if="s.status === 'running'" class="spinner" />
        <span class="step-label">{{ s.label }}</span>
        <span v-if="s.detail" class="step-detail">{{ s.detail }}</span>
      </span>
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
.chat-window { height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 8px; background: #fafafa; }
.msg { margin-bottom: 12px; padding: 8px 12px; border-radius: 8px; max-width: 80%; }
.msg.user { background: #409EFF; color: white; margin-left: auto; }
.msg.assistant { background: #e8e8e8; }
.product-cards { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.product-card { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 8px 10px; min-width: 130px; }
.p-name { font-size: 13px; font-weight: 600; color: #333; }
.p-meta { font-size: 12px; color: #888; margin: 2px 0 6px; }
.p-btn { padding: 3px 10px; background: #409EFF; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }
.p-btn:hover { background: #337ecc; }
.cart-feedback { margin-bottom: 8px; padding: 6px 12px; background: #f0f9eb; border: 1px solid #c2e7b0; border-radius: 6px; font-size: 13px; color: #529b2e; }
.steps { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
.step { display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px; border-radius: 14px; font-size: 12px; background: #f0f4ff; color: #4066c0; }
.step.running { background: #e8f3ff; }
.step.done { opacity: .65; }
.step-detail { color: #666; max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.spinner { width: 10px; height: 10px; border: 2px solid #b0c4ff; border-top-color: #4066c0; border-radius: 50%; animation: spin .8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.chat-input { display: flex; flex-direction: column; gap: 10px; }
.input-row { display: flex; gap: 12px; }
.input-row input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
.input-row button { padding: 10px 24px; background: #409EFF; color: white; border: none; border-radius: 4px; cursor: pointer; }
.quick-actions { display: flex; gap: 8px; }
.quick-btn { padding: 6px 14px; background: #f0f0f0; border: 1px solid #ddd; border-radius: 20px; cursor: pointer; font-size: 13px; transition: background .2s; }
.quick-btn:hover { background: #e0e0e0; }
</style>
