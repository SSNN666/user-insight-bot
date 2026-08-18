/** API 客户端测试:SSE 流式解析(零依赖手写帧拆分)。 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { askAgentStream } from '../index'

function mockFetchSSE(chunks: string[]) {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      chunks.forEach(c => controller.enqueue(new TextEncoder().encode(c)))
      controller.close()
    },
  })
  return vi.fn().mockResolvedValue(new Response(body, { status: 200 }))
}

afterEach(() => vi.restoreAllMocks())

describe('askAgentStream', () => {
  it('解析多帧 SSE 事件', async () => {
    vi.stubGlobal('fetch', mockFetchSSE([
      'data: {"event":"meta","session_id":"s"}\n\n',
      'data: {"event":"delta","content":"你好"}\n\n',
      'data: {"event":"done","reply":"你好"}\n\n',
    ]))
    const events: any[] = []
    await askAgentStream('问题', 's', ev => events.push(ev))
    expect(events.map(e => e.event)).toEqual(['meta', 'delta', 'done'])
    expect(events[2].reply).toBe('你好')
  })

  it('跨 chunk 的帧正确重组(缓冲逻辑)', async () => {
    // 帧被切成两半,分别在两个 chunk 中
    vi.stubGlobal('fetch', mockFetchSSE([
      'data: {"event":"delta","con',
      'tent":"跨块"}\n\ndata: {"event":"done","reply":"ok"}\n\n',
    ]))
    const events: any[] = []
    await askAgentStream('q', 's', ev => events.push(ev))
    expect(events).toHaveLength(2)
    expect(events[0].content).toBe('跨块')
  })

  it('忽略坏帧,不中断后续解析', async () => {
    vi.stubGlobal('fetch', mockFetchSSE([
      'data: {not-json\n\n',
      'data: {"event":"done","reply":"ok"}\n\n',
    ]))
    const events: any[] = []
    await askAgentStream('q', 's', ev => events.push(ev))
    expect(events.map(e => e.event)).toEqual(['done'])
  })

  it('HTTP 错误 → error 事件', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 500 })))
    const events: any[] = []
    await askAgentStream('q', 's', ev => events.push(ev))
    expect(events[0].event).toBe('error')
  })
})
