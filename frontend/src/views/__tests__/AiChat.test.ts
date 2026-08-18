/** AiChat 对话组件:流式事件驱动渲染(SSE 步骤条/商品卡片/图表)。 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import AiChat from '../AiChat.vue'

const addToCartMock = vi.fn().mockResolvedValue({})
const streamMock = vi.fn()

vi.mock('../../api', () => ({
  askAgentStream: (...args: any[]) => streamMock(...args),
  addToCart: (...args: any[]) => addToCartMock(...args),
}))

/** 让 mock 流依次回放事件(可断言调用参数)。 */
function playEvents(events: any[]) {
  streamMock.mockImplementationOnce((_q, _sid, onEvent) => {
    events.forEach(ev => onEvent(ev))
    return Promise.resolve()
  })
}

beforeEach(() => {
  streamMock.mockReset()
  addToCartMock.mockClear()
})

describe('AiChat', () => {
  it('挂载后展示欢迎消息', async () => {
    const w = mount(AiChat)
    await flushPromises()
    const texts = w.findAll('.msg.assistant').map(m => m.text())
    expect(texts.some(t => t.includes('你好'))).toBe(true)
  })

  it('发送问题 → 用户消息 + AI 流式回答累积', async () => {
    playEvents([
      { event: 'node_start', node: 'preprocess' },
      { event: 'delta', content: '正在分析' },
      { event: 'delta', content: '分群数据' },
      { event: 'done', reply: '正在分析分群数据', tools_called: [], products: [], charts: [] },
    ])
    const w = mount(AiChat)
    await w.find('input').setValue('分群数据怎么样')
    await w.find('.input-row button').trigger('click')
    await flushPromises()

    const texts = w.findAll('.msg').map(m => m.text())
    expect(texts.some(t => t.includes('你:') && t.includes('分群数据怎么样'))).toBe(true)
    expect(texts.some(t => t.includes('AI:') && t.includes('正在分析分群数据'))).toBe(true)
    // mock 流收到问题与 session
    expect(streamMock).toHaveBeenCalledWith('分群数据怎么样', 'vue-chat', expect.any(Function), expect.anything())
  })

  it('node_start → 步骤条渲染;done → 步骤全部完成', async () => {
    playEvents([
      { event: 'node_start', node: 'preprocess' },
      { event: 'node_start', node: 'llm_decide' },
      { event: 'done', reply: 'ok', tools_called: [], products: [], charts: [] },
    ])
    const w = mount(AiChat)
    await w.find('input').setValue('分析')
    await w.find('.input-row button').trigger('click')
    await flushPromises()

    const labels = w.findAll('.step-label').map(s => s.text())
    expect(labels).toContain('🔍 意图识别')
    expect(labels).toContain('🤖 模型决策')
    expect(w.findAll('.step.done')).toHaveLength(2)   // 全部完成
  })

  it('done 事件携带商品 → 渲染商品卡片,点击加购', async () => {
    playEvents([{
      event: 'done', reply: '推荐以下商品', tools_called: [],
      products: [{ product_id: 166345, product_name: 'SKU-166345', category: '8', price: 4110 }],
      charts: [],
    }])
    const w = mount(AiChat)
    await w.find('input').setValue('推荐')
    await w.find('.input-row button').trigger('click')
    await flushPromises()

    expect(w.text()).toContain('SKU-166345')
    await w.find('.p-btn').trigger('click')
    await flushPromises()
    expect(addToCartMock).toHaveBeenCalledWith(166345, 1)
    expect(w.text()).toContain('已加入购物车')
  })

  it('done 事件携带图表 → 渲染 MiniChart', async () => {
    playEvents([{
      event: 'done', reply: '分群统计', tools_called: [],
      products: [],
      charts: [{ type: 'bar', title: '各分群人数', categories: ['分群0'], series: [{ name: '用户数', data: [100] }] }],
    }])
    const w = mount(AiChat)
    await w.find('input').setValue('分群')
    await w.find('.input-row button').trigger('click')
    await flushPromises()

    expect(w.text()).toContain('各分群人数')
    expect(w.findAll('rect').length).toBeGreaterThan(0)
  })

  it('error 事件 → 兜底错误文案', async () => {
    playEvents([{ event: 'error', message: '服务器内部错误' }])
    const w = mount(AiChat)
    await w.find('input').setValue('测试')
    await w.find('.input-row button').trigger('click')
    await flushPromises()

    const texts = w.findAll('.msg.assistant').map(m => m.text())
    expect(texts.some(t => t.includes('服务器内部错误'))).toBe(true)
  })

  it('loading 期间重复点击发送被忽略', async () => {
    streamMock.mockImplementationOnce((_q, _sid, _onEvent) => new Promise(() => {}))  // 挂起
    const w = mount(AiChat)
    await w.find('input').setValue('问题')
    await w.find('.input-row button').trigger('click')
    await w.find('input').setValue('另一个')
    await w.find('.input-row button').trigger('click')
    await flushPromises()
    expect(streamMock).toHaveBeenCalledTimes(1)
  })
})
