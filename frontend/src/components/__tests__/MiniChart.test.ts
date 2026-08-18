/** MiniChart:零依赖 SVG 图表渲染(bar / line)。 */
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import MiniChart from '../MiniChart.vue'

const BAR_CHART = {
  type: 'bar' as const,
  title: '转化漏斗(浏览→加购→下单)',
  categories: ['浏览', '加购', '下单'],
  series: [
    { name: '用户数', data: [22833, 4511, 790] },
    { name: '转化率(%)', data: [100, 19.8, 17.5] },
  ],
}

const LINE_CHART = {
  type: 'line' as const,
  title: '分群人数时间趋势',
  categories: ['02-01', '02-02'],
  series: [
    { name: '分群0', data: [60, 61] },
    { name: '分群2', data: [10, 11] },
  ],
}

describe('MiniChart', () => {
  it('bar 图:渲染标题、类别数 × 系列数的矩形、图例', () => {
    const w = mount(MiniChart, { props: { chart: BAR_CHART } })
    expect(w.text()).toContain('转化漏斗(浏览→加购→下单)')
    // 3 类别 × 2 系列数据柱 + 2 个图例色块 = 8 个矩形
    expect(w.findAll('rect')).toHaveLength(8)
    // 类别标签
    expect(w.text()).toContain('浏览')
    expect(w.text()).toContain('下单')
    // 图例
    expect(w.text()).toContain('用户数')
    expect(w.text()).toContain('转化率(%)')
  })

  it('line 图:每个系列一条折线 + 数据点', () => {
    const w = mount(MiniChart, { props: { chart: LINE_CHART } })
    expect(w.text()).toContain('分群人数时间趋势')
    expect(w.findAll('polyline')).toHaveLength(2)   // 2 个系列
    expect(w.text()).toContain('分群0')
    expect(w.text()).toContain('分群2')
  })

  it('空数据不崩溃', () => {
    const w = mount(MiniChart, {
      props: { chart: { type: 'bar', title: '空', categories: [], series: [] } },
    })
    expect(w.exists()).toBe(true)
  })
})
