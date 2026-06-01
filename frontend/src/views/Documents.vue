<template>
  <div class="docs-page">
    <div class="toolbar">
      <el-button :icon="ArrowLeft" @click="router.back()">返回</el-button>
      <el-upload
        :show-file-list="false"
        :before-upload="beforeUpload"
        :http-request="handleUpload"
        multiple
        accept=".pdf,.docx,.doc,.xlsx,.xls,.txt,.md,.csv,.png,.jpg,.jpeg,.gif,.bmp,.webp"
      >
        <el-button type="primary" :icon="Upload">上传文档</el-button>
      </el-upload>
      <span class="upload-hint">支持 PDF、Word、Excel、TXT、Markdown、图片（PNG/JPG/GIF/BMP/WebP）</span>
    </div>

    <el-table v-loading="loading" :data="docList" border stripe>
      <el-table-column prop="filename" label="文件名" min-width="200" show-overflow-tooltip />
      <el-table-column prop="file_type" label="类型" width="80">
        <template #default="{ row }">
          <el-tag size="small" :type="isImage(row.file_type) ? 'warning' : 'primary'">
            {{ row.file_type.toUpperCase() }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="file_size" label="大小" width="90">
        <template #default="{ row }">{{ formatSize(row.file_size) }}</template>
      </el-table-column>
      <el-table-column prop="status" label="状态" width="120">
        <template #default="{ row }">
          <el-popover
            v-if="row.status === 'failed' && row.error_msg"
            placement="top"
            :width="320"
            trigger="hover"
          >
            <template #reference>
              <el-tag type="danger" size="small" class="error-tag">失败 <el-icon><WarningFilled /></el-icon></el-tag>
            </template>
            <div class="error-msg-box">
              <div class="error-msg-title">失败原因</div>
              <div class="error-msg-content">{{ row.error_msg }}</div>
            </div>
          </el-popover>
          <el-tag v-else :type="statusType(row.status)" size="small">{{ statusLabel(row.status) }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="chunk_count" label="分块数" width="80" />
      <el-table-column prop="tags" label="标签" width="120" show-overflow-tooltip />
      <el-table-column prop="created_at" label="上传时间" width="160">
        <template #default="{ row }">{{ formatDate(row.created_at) }}</template>
      </el-table-column>
      <el-table-column label="操作" width="200" fixed="right">
        <template #default="{ row }">
          <el-button v-if="isImage(row.file_type)" size="small" type="primary" link @click="previewImage(row)">
            预览
          </el-button>
          <el-button size="small" link @click="downloadFile(row)">下载</el-button>
          <el-button v-if="row.status === 'failed'" size="small" @click="reprocess(row.id)">重试</el-button>
          <el-button size="small" type="danger" link @click="deleteDoc(row)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-pagination
      v-model:current-page="page"
      v-model:page-size="pageSize"
      :total="total"
      layout="total, prev, pager, next"
      class="pagination"
      @change="loadDocs"
    />

    <!-- 图片预览弹窗 -->
    <el-dialog v-model="previewVisible" title="图片预览" width="800px" destroy-on-close>
      <div class="preview-container">
        <img v-if="previewUrl" :src="previewUrl" :alt="previewFilename" class="preview-image" />
        <div v-if="previewFilename" class="preview-filename">{{ previewFilename }}</div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { ArrowLeft, Upload, WarningFilled } from '@element-plus/icons-vue'
import { docsApi } from '@/api'

const IMAGE_TYPES = new Set(['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'])
const DOC_MAX_SIZE = 100 * 1024 * 1024
const IMAGE_MAX_SIZE = 10 * 1024 * 1024

const route = useRoute()
const router = useRouter()
const kbId = Number(route.params.kbId)

const docList = ref([])
const loading = ref(false)
const page = ref(1)
const pageSize = ref(20)
const total = ref(0)
const previewVisible = ref(false)
const previewUrl = ref('')
const previewFilename = ref('')
let pollTimer = null

function isImage(ext) { return IMAGE_TYPES.has((ext || '').toLowerCase()) }

async function loadDocs() {
  loading.value = true
  try {
    const res = await docsApi.list({ kb_id: kbId, page: page.value, page_size: pageSize.value })
    docList.value = res.data || []
    total.value = res.total || 0
  } finally {
    loading.value = false
  }
}

function beforeUpload(file) {
  const ext = file.name.split('.').pop()?.toLowerCase() || ''
  const isImg = IMAGE_TYPES.has(ext)
  const maxSize = isImg ? IMAGE_MAX_SIZE : DOC_MAX_SIZE
  if (file.size > maxSize) {
    ElMessage.error(`文件超过 ${maxSize / 1024 / 1024}MB 限制`)
    return false
  }
  return true
}

async function handleUpload({ file }) {
  const token = localStorage.getItem('token')
  const worker = new Worker(
    new URL('../workers/upload.worker.js', import.meta.url),
    { type: 'module' }
  )

  worker.postMessage({
    file,
    kbId,
    token,
    uploadUrl: '/api/v1/docs/upload',
  })

  worker.addEventListener('message', ({ data }) => {
    worker.terminate()
    if (data.success && data.data?.data) {
      page.value = 1
      docList.value = [data.data.data, ...docList.value]
      total.value++
      ElMessage.success(`${file.name} 上传成功，后台处理中`)
    } else if (data.status === 409) {
      ElMessage.warning(data.error || `知识库内已存在同名文件「${file.name}」`)
    } else {
      ElMessage.error(data.error || '上传失败')
    }
  }, { once: true })

  worker.addEventListener('error', () => {
    worker.terminate()
    ElMessage.error('上传失败，请检查后端服务是否正常运行')
  }, { once: true })
}

async function deleteDoc(row) {
  await ElMessageBox.confirm(`确定删除文档「${row.filename}」？`, '警告', { type: 'warning' })
  await docsApi.delete(row.id)
  ElMessage.success('已删除')
  loadDocs()
}

async function reprocess(id) {
  await docsApi.reprocess(id)
  ElMessage.success('已重新提交处理')
  loadDocs()
}

function previewImage(row) {
  previewUrl.value = `/api/v1/docs/${row.id}/file`
  previewFilename.value = row.filename
  previewVisible.value = true
}

function downloadFile(row) {
  const url = `/api/v1/docs/${row.id}/file`
  const link = document.createElement('a')
  link.href = url
  link.download = row.filename
  link.target = '_blank'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

function statusType(s) { return { completed: 'success', failed: 'danger', processing: 'warning', pending: 'info' }[s] || 'info' }
function statusLabel(s) { return { completed: '已完成', failed: '失败', processing: '处理中', pending: '待处理' }[s] || s }
function formatSize(bytes) {
  if (!bytes) return '-'
  if (bytes < 1024) return bytes + 'B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + 'KB'
  return (bytes / 1024 / 1024).toFixed(1) + 'MB'
}
function formatDate(str) { return str ? new Date(str).toLocaleString() : '' }

onMounted(loadDocs)
pollTimer = setInterval(() => {
  if (docList.value.some(d => d.status === 'processing' || d.status === 'pending')) loadDocs()
}, 5000)
onBeforeUnmount(() => clearInterval(pollTimer))
</script>

<style scoped>
.docs-page { padding: 20px; }
.toolbar { display: flex; gap: 12px; margin-bottom: 16px; align-items: center; flex-wrap: wrap; }
.upload-hint { font-size: 12px; color: #909399; }
.pagination { margin-top: 16px; justify-content: flex-end; display: flex; }
.preview-container { text-align: center; }
.preview-image { max-width: 100%; max-height: 70vh; border-radius: 8px; object-fit: contain; }
.preview-filename { margin-top: 10px; font-size: 13px; color: #909399; }

.error-tag { cursor: pointer; }
.error-tag:hover { opacity: 0.8; }
.error-msg-box { font-size: 13px; line-height: 1.6; }
.error-msg-title { font-weight: 600; margin-bottom: 6px; color: var(--el-color-danger); }
.error-msg-content { color: #303133; word-break: break-all; }
</style>
