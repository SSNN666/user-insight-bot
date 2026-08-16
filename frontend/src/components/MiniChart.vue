<script setup lang="ts">
// 零依赖内联 SVG 图表(bar / line),协议见 agent/chart_extract.py:
// { type, title, categories, series: [{name, data}] }
interface Series { name: string; data: (number | null)[] }
interface Chart { type: 'bar' | 'line'; title: string; categories: string[]; series: Series[] }
const props = defineProps<{ chart: Chart }>()

const COLORS = ['#409EFF', '#67C23A', '#E6A23C', '#F56C6C', '#8B5CF6', '#13C2C2', '#fa8c16', '#722ed1']
const W = 460, H = 250, PAD_L = 44, PAD_R = 12, PAD_T = 36, PAD_B = 60
const innerW = W - PAD_L - PAD_R
const innerH = H - PAD_T - PAD_B

const allVals = props.chart.series.flatMap(s => s.data).filter((v): v is number => v !== null)
const maxVal = Math.max(...allVals, 1)
const y = (v: number) => PAD_T + innerH - (v / maxVal) * innerH
const ticks = [0, 0.5, 1].map(t => ({ v: t * maxVal, y: y(t * maxVal) }))

const n = props.chart.categories.length
const groupW = innerW / Math.max(n, 1)
const barW = (groupW * 0.72) / Math.max(props.chart.series.length, 1)
const xStep = innerW / Math.max(n - 1, 1)

function linePoints(s: Series): string {
  const pts = s.data
    .map((v, i) => (v === null ? null : `${(PAD_L + i * xStep).toFixed(1)},${y(v).toFixed(1)}`))
    .filter((p): p is string => p !== null)
  return pts.join(' ')
}

// x 轴标签:超过 8 个类别时隔一显示,避免重叠
const labelEvery = n > 8 ? Math.ceil(n / 8) : 1
const labelX = (i: number) => PAD_L + i * xStep
const fmt = (v: number) =>
  v >= 10000 ? `${(v / 10000).toFixed(1)}万`
    : v >= 1000 ? `${(v / 1000).toFixed(1)}k`
      : String(Math.round(v * 10) / 10)
</script>

<template>
  <div class="mini-chart">
    <div class="mc-title">{{ chart.title }}</div>
    <svg :width="W" :height="H" :viewBox="`0 0 ${W} ${H}`">
      <!-- 网格与 y 轴刻度 -->
      <line v-for="t in ticks" :key="t.v" :x1="PAD_L" :x2="W - PAD_R" :y1="t.y" :y2="t.y" stroke="#eee" />
      <text v-for="t in ticks" :key="'l' + t.v" :x="PAD_L - 6" :y="t.y + 3" text-anchor="end" font-size="9" fill="#999">{{ fmt(t.v) }}</text>

      <!-- bar -->
      <template v-if="chart.type === 'bar'">
        <g v-for="(cat, gi) in chart.categories" :key="cat">
          <rect v-for="(s, si) in chart.series" :key="s.name"
            :x="PAD_L + gi * groupW + groupW * 0.14 + si * (barW * 1.05)"
            :y="y(s.data[gi] ?? 0)" :width="barW"
            :height="Math.max(0, PAD_T + innerH - y(s.data[gi] ?? 0))"
            :fill="COLORS[si % COLORS.length]" rx="2" />
        </g>
        <text v-for="(cat, gi) in chart.categories" :key="'x' + gi"
          v-show="gi % labelEvery === 0"
          :x="PAD_L + gi * groupW + groupW / 2" :y="H - 42"
          text-anchor="middle" font-size="9" fill="#666">{{ cat }}</text>
      </template>

      <!-- line -->
      <template v-else>
        <polyline v-for="(s, si) in chart.series" :key="s.name"
          :points="linePoints(s)" fill="none" :stroke="COLORS[si % COLORS.length]"
          stroke-width="2" stroke-linejoin="round" />
        <g v-for="(s, si) in chart.series" :key="'dots' + s.name">
          <circle v-for="(v, i) in s.data" :key="i"
            v-show="v !== null && i % labelEvery === 0"
            :cx="labelX(i)" :cy="y(v ?? 0)" r="3" :fill="COLORS[si % COLORS.length]" />
        </g>
        <text v-for="(cat, gi) in chart.categories" :key="'x' + gi"
          v-show="gi % labelEvery === 0"
          :x="labelX(gi)" :y="H - 42" text-anchor="middle" font-size="9" fill="#666"
          :transform="`rotate(-25, ${labelX(gi)}, ${H - 42})`">{{ cat }}</text>
      </template>

      <!-- 图例(两行,最多 8 系列) -->
      <g v-for="(s, si) in chart.series" :key="'leg' + s.name">
        <rect :x="PAD_L + (si % 4) * 110" :y="H - 26 + Math.floor(si / 4) * 16"
          width="10" height="10" rx="2" :fill="COLORS[si % COLORS.length]" />
        <text :x="PAD_L + (si % 4) * 110 + 14" :y="H - 17 + Math.floor(si / 4) * 16"
          font-size="10" fill="#555">{{ s.name }}</text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.mini-chart { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px 12px; margin-top: 10px; max-width: 100%; overflow-x: auto; }
.mc-title { font-size: 13px; font-weight: 600; color: #333; margin-bottom: 4px; }
svg { display: block; }
</style>
