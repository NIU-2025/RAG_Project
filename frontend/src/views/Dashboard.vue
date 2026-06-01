<template>
  <div class="dashboard">
    <el-row :gutter="16" class="stat-cards">
      <el-col :span="6">
        <el-card class="stat-card" shadow="hover">
          <div class="stat-icon" style="background:#409EFF20;color:#409EFF">
            <el-icon><FolderOpened /></el-icon>
          </div>
          <div class="stat-body">
            <div class="stat-value">{{ overview.total_kbs }}</div>
            <div class="stat-label">知识库总数</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card" shadow="hover">
          <div class="stat-icon" style="background:#67C23A20;color:#67C23A">
            <el-icon><Document /></el-icon>
          </div>
          <div class="stat-body">
            <div class="stat-value">{{ overview.total_docs }}</div>
            <div class="stat-label">文档总数</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card" shadow="hover">
          <div class="stat-icon" style="background:#E6A23C20;color:#E6A23C">
            <el-icon><DataBoard /></el-icon>
          </div>
          <div class="stat-body">
            <div class="stat-value">{{ overview.total_size_display }}</div>
            <div class="stat-label">总大小</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card" shadow="hover">
          <div class="stat-icon" style="background:#F56C6C20;color:#F56C6C">
            <el-icon><ChatLineSquare /></el-icon>
          </div>
          <div class="stat-body">
            <div class="stat-value">{{ parseStats.total }}</div>
            <div class="stat-label">解析任务总数</div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 文档解析成功率 & 每日问答趋势 -->
    <el-row :gutter="16" class="chart-row">
      <el-col :span="12">
        <el-card class="chart-card">
          <template #header>
            <div class="card-title">文档解析情况</div>
          </template>
          <div v-if="parseStats.total > 0" class="parse-stats">
            <div class="rate-row">
              <el-progress
                type="circle"
                :percentage="Math.round(parseStats.success_rate * 100)"
                :color="successColor"
                :width="120"
              >
                <template #default>
                  <span class="circle-rate">{{ Math.round(parseStats.success_rate * 100) }}%</span>
                  <span class="circle-sub">成功率</span>
                </template>
              </el-progress>
              <div class="rate-detail">
                <div class="rate-item">
                  <span class="dot success-dot"></span>
                  <span class="rate-label">成功</span>
                  <span class="rate-val">{{ parseStats.success }}</span>
                </div>
                <div class="rate-item">
                  <span class="dot failed-dot"></span>
                  <span class="rate-label">失败</span>
                  <span class="rate-val">{{ parseStats.failed }}</span>
                </div>
                <div class="rate-item">
                  <span class="dot pending-dot"></span>
                  <span class="rate-label">处理中</span>
                  <span class="rate-val">{{ parseStats.pending }}</span>
                </div>
                <el-progress :percentage="Math.round(parseStats.failed_rate * 100)" :color="failColor" :stroke-width="8" style="margin-top:8px" />
              </div>
            </div>
          </div>
          <el-empty v-else description="暂无文档数据" :image-size="80" />
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card class="chart-card">
          <template #header>
            <div class="card-title">每日问答趋势（最近30天）</div>
          </template>
          <div v-if="dailyChat.length > 0" class="chart-wrap">
            <div ref="chatChartRef" class="chart-container"></div>
          </div>
          <el-empty v-else description="暂无问答数据" :image-size="80" />
        </el-card>
      </el-col>
    </el-row>

    <!-- 知识库统计 -->
    <el-card class="kb-stats-card">
      <template #header>
        <div class="card-title">各知识库统计</div>
      </template>
      <el-table :data="kbStats" stripe v-loading="loadingKb">
        <el-table-column label="知识库" min-width="160">
          <template #default="{ row }">
            <div class="kb-name-cell">
              <el-icon style="color:#409EFF;margin-right:6px"><FolderOpened /></el-icon>
              {{ row.kb_name }}
            </div>
          </template>
        </el-table-column>
        <el-table-column label="文档数" prop="doc_count" width="100" align="center" />
        <el-table-column label="总大小" width="120" align="center">
          <template #default="{ row }">{{ row.total_size_display }}</template>
        </el-table-column>
        <el-table-column label="成功" width="80" align="center">
          <template #default="{ row }">
            <el-tag size="small" type="success">{{ row.success_count }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="失败" width="80" align="center">
          <template #default="{ row }">
            <el-tag size="small" type="danger">{{ row.failed_count }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="待处理" width="90" align="center">
          <template #default="{ row }">
            <el-tag size="small" type="info">{{ row.pending_count }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="各类型占比" min-width="220">
          <template #default="{ row }">
            <div class="type-bars" v-if="Object.keys(row.file_type_breakdown).length">
              <div v-for="(item, type) in row.file_type_breakdown" :key="type" class="type-bar-row">
                <span class="type-name">{{ type.toUpperCase() }}</span>
                <span class="type-badge">{{ item.count }} 份</span>
                <el-progress
                  :percentage="Math.round((item.count / row.doc_count) * 100)"
                  :color="typeColor(type)"
                  :stroke-width="8"
                  :show-text="false"
                  style="flex:1;min-width:40px"
                />
                <span class="type-size">{{ item.size_display }}</span>
              </div>
            </div>
            <span v-else class="no-data">-</span>
          </template>
        </el-table-column>
      </el-table>
      <el-empty v-if="!loadingKb && kbStats.length === 0" description="暂无知识库" :image-size="60" />
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, nextTick, onUnmounted } from 'vue'
import { statsApi } from '@/api'
import { FolderOpened, Document, DataBoard, ChatLineSquare } from '@element-plus/icons-vue'
import * as echarts from 'echarts'

const overview = ref({ total_kbs: 0, total_docs: 0, total_size_display: '0 B', total_chunks: 0 })
const parseStats = ref({ total: 0, success: 0, failed: 0, pending: 0, success_rate: 0, failed_rate: 0 })
const kbStats = ref([])
const dailyChat = ref([])
const loadingKb = ref(false)
const chatChartRef = ref(null)
let chatChart = null

const successColor = [{ color: '#67C23A', percentage: 100 }]
const failColor = [{ color: '#F56C6C', percentage: 100 }]

const TYPE_COLORS = {
  pdf: '#E6A23C', docx: '#409EFF', doc: '#409EFF', txt: '#909399',
  xlsx: '#67C23A', xls: '#67C23A', csv: '#67C23A',
  png: '#9B59B6', jpg: '#9B59B6', jpeg: '#9B59B6', gif: '#9B59B6',
  mp3: '#F56C6C', wav: '#F56C6C', m4a: '#F56C6C',
  unknown: '#C0C4CC',
}
function typeColor(type) {
  return TYPE_COLORS[type?.toLowerCase()] || '#409EFF'
}

async function loadAll() {
  const [ov, ps, kb, chat] = await Promise.all([
    statsApi.overview().catch(() => ({})),
    statsApi.parse().catch(() => ({ total: 0 })),
    statsApi.kbs().catch(() => []),
    statsApi.chatDaily(30).catch(() => []),
  ])
  overview.value = ov
  parseStats.value = ps
  kbStats.value = kb
  dailyChat.value = chat
  await nextTick()
  if (chatChartRef.value && chat.length > 0) renderChatChart(chat)
}

function renderChatChart(data) {
  if (!chatChartRef.value) return
  if (chatChart) { chatChart.dispose() }
  chatChart = echarts.init(chatChartRef.value)
  const dates = data.map(d => d.date?.slice(5) || '')
  const sessions = data.map(d => d.sessions)
  const messages = data.map(d => d.messages)
  chatChart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['会话数', '消息数'], bottom: 0 },
    grid: { left: 40, right: 20, top: 10, bottom: 40 },
    xAxis: { type: 'category', data: dates, axisLabel: { fontSize: 11 } },
    yAxis: { type: 'value', minInterval: 1 },
    series: [
      { name: '会话数', type: 'line', data: sessions, smooth: true, color: '#409EFF', areaStyle: { color: 'rgba(64,158,255,0.1)' } },
      { name: '消息数', type: 'line', data: messages, smooth: true, color: '#67C23A', areaStyle: { color: 'rgba(103,194,58,0.1)' } },
    ],
  })
}

function handleResize() { chatChart?.resize() }

onMounted(() => {
  loadAll()
  window.addEventListener('resize', handleResize)
})
onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  chatChart?.dispose()
})
</script>

<style scoped>
.dashboard { padding: 20px; }
.stat-cards { margin-bottom: 16px; }
.stat-card { display: flex; align-items: center; gap: 16px; padding: 8px 0; }
.stat-icon { width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 22px; flex-shrink: 0; }
.stat-body { flex: 1; }
.stat-value { font-size: 24px; font-weight: 700; color: #303133; line-height: 1.2; }
.stat-label { font-size: 13px; color: #909399; margin-top: 4px; }
.chart-row { margin-bottom: 16px; }
.chart-card { height: 280px; }
.card-title { font-size: 15px; font-weight: 600; color: #303133; }
.parse-stats { padding: 8px 0; }
.rate-row { display: flex; align-items: center; gap: 32px; }
.circle-rate { display: block; font-size: 20px; font-weight: 700; color: #67C23A; line-height: 1.2; }
.circle-sub { display: block; font-size: 12px; color: #909399; }
.rate-detail { flex: 1; }
.rate-item { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.dot { width: 8px; height: 8px; border-radius: 50%; }
.success-dot { background: #67C23A; }
.failed-dot { background: #F56C6C; }
.pending-dot { background: #909399; }
.rate-label { font-size: 13px; color: #606266; width: 50px; }
.rate-val { font-size: 13px; font-weight: 600; color: #303133; }
.chart-wrap { height: 210px; }
.chart-container { width: 100%; height: 100%; }
.kb-stats-card { margin-bottom: 16px; }
.kb-name-cell { display: flex; align-items: center; font-weight: 500; }
.type-bars { display: flex; flex-direction: column; gap: 6px; }
.type-bar-row { display: flex; align-items: center; gap: 8px; }
.type-name { font-size: 12px; color: #606266; width: 36px; flex-shrink: 0; font-family: monospace; }
.type-badge { font-size: 12px; font-weight: 600; color: #303133; width: 48px; flex-shrink: 0; text-align: center; }
.type-size { font-size: 12px; color: #909399; width: 72px; text-align: right; flex-shrink: 0; }
.no-data { color: #C0C4CC; }
</style>
