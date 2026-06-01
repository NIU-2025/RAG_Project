<template>
  <div class="chat-page">
    <!-- 左侧：会话列表 -->
    <div class="session-panel">
      <div class="session-header">
        <span>历史对话</span>
        <el-button :icon="Plus" circle size="small" @click="showNewSession = true" />
      </div>
      <div class="session-list">
        <div
          v-for="s in sessions"
          :key="s.id"
          :class="['session-item', { active: currentSession?.id === s.id }]"
          @click="switchSession(s)"
        >
          <div class="session-title">{{ s.title }}</div>
          <div class="session-meta">{{ s.llm_provider }} · {{ formatDate(s.created_at) }}</div>
          <el-icon class="del-icon" @click.stop="deleteSession(s)"><Delete /></el-icon>
        </div>
      </div>
    </div>

    <!-- 右侧：对话区 -->
    <div class="chat-area">
      <div v-if="!currentSession" class="no-session">
        <el-empty description="请选择或新建一个对话" />
      </div>

      <template v-else>
        <!-- 消息列表 -->
        <div class="messages" ref="messagesEl">
          <div v-for="msg in messages" :key="msg.id || msg._tmpId" :class="['message', msg.role]">
            <div class="msg-content" v-html="renderMd(msg.content)" />
            <div v-if="msg.sources && parseSources(msg.sources).length" class="sources">
              <div class="sources-title">
                <el-icon><DocumentCopy /></el-icon>
                引用来源（{{ parseSources(msg.sources).length }} 条，按相关度排序）
              </div>
              <div v-for="(src, i) in parseSources(msg.sources)" :key="i" class="source-card">
                <div class="source-header">
                  <el-tag size="small" type="primary">[{{ src.citation_index ?? i + 1 }}]</el-tag>
                  <span class="source-filename">{{ src.filename }}</span>
                  <span class="source-page">{{ isImageType(src.file_type) ? '图片' : `第 ${src.page_num || '-'} 页` }}</span>
                  <el-tag size="small" type="success" class="source-score">
                    相关度 {{ (src.score * 100).toFixed(0) }}%
                  </el-tag>
                  <el-button
                    v-if="isImageType(src.file_type)"
                    size="small"
                    type="primary"
                    link
                    @click="previewImage(src)"
                    style="margin-left:4px"
                  >
                    <el-icon><Picture /></el-icon> 查看图片
                  </el-button>
                </div>
                <div v-if="isImageType(src.file_type)" class="source-thumbnail-wrapper">
                  <img
                    :src="`/api/v1/docs/${src.doc_id}/file`"
                    :alt="src.filename"
                    class="source-thumbnail"
                    @click="previewImage(src)"
                    @error="onThumbnailError"
                  />
                </div>
                <div v-if="!isImageType(src.file_type)" class="source-content">{{ src.content }}</div>
              </div>
            </div>
          </div>

          <div v-if="streaming" class="message assistant">
            <div class="msg-content" v-html="renderMd(streamBuffer)" />
            <span class="cursor">▋</span>
          </div>
        </div>

        <!-- 输入框 -->
        <div class="input-area">
          <el-input
            v-model="inputText"
            type="textarea"
            :rows="3"
            placeholder="输入问题，Enter 发送，Shift+Enter 换行"
            resize="none"
            :disabled="streaming"
            @keydown="onInputKeydown"
          />
          <div class="input-actions">
            <el-select v-model="currentSession.llm_provider" size="small" style="width:140px">
              <el-option label="OpenAI" value="openai" />
              <el-option label="DeepSeek" value="deepseek" />
              <el-option label="通义千问" value="dashscope" />
              <el-option label="文心一言" value="qianfan" />
              <el-option label="Ollama" value="ollama" />
              <el-option label="LM Studio" value="lmstudio" />
            </el-select>
            <el-button type="primary" :disabled="streaming || !inputText.trim()" @click="sendMessage">
              {{ streaming ? '生成中...' : '发送' }}
            </el-button>
          </div>
        </div>
      </template>
    </div>

    <!-- 新建会话弹窗 -->
    <el-dialog v-model="showNewSession" title="新建对话" width="400px">
      <el-form :model="newSessionForm" label-position="top">
        <el-form-item label="关联知识库">
          <el-select v-model="newSessionForm.kb_id" placeholder="选择知识库（可选）" clearable style="width:100%">
            <el-option v-for="kb in kbList" :key="kb.id" :label="kb.name" :value="kb.id" />
          </el-select>
        </el-form-item>
            <el-form-item label="大模型">
          <el-select v-model="newSessionForm.llm_provider" style="width:100%">
            <el-option label="OpenAI" value="openai" />
            <el-option label="DeepSeek" value="deepseek" />
            <el-option label="通义千问" value="dashscope" />
            <el-option label="文心一言" value="qianfan" />
            <el-option label="Ollama (本地)" value="ollama" />
            <el-option label="LM Studio (本地)" value="lmstudio" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showNewSession = false">取消</el-button>
        <el-button type="primary" @click="createSession">创建</el-button>
      </template>
    </el-dialog>

    <!-- 图片预览弹窗 -->
    <el-dialog v-model="previewVisible" :title="previewFilename" width="800px" destroy-on-close>
      <div class="preview-container">
        <img v-if="previewUrl" :src="previewUrl" :alt="previewFilename" class="preview-image" />
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Plus, Delete, DocumentCopy, Picture } from '@element-plus/icons-vue'
import { marked } from 'marked'
import { chatApi, kbApi } from '@/api'

const IMAGE_TYPES = new Set(['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'])
function isImageType(ext) { return IMAGE_TYPES.has((ext || '').toLowerCase()) }

const route = useRoute()
const sessions = ref([])
const currentSession = ref(null)
const messages = ref([])
const inputText = ref('')
const streaming = ref(false)
const streamBuffer = ref('')
const messagesEl = ref(null)
const showNewSession = ref(false)
const kbList = ref([])
const newSessionForm = reactive({ kb_id: null, llm_provider: 'openai', title: '' })
const previewVisible = ref(false)
const previewUrl = ref('')
const previewFilename = ref('')

marked.setOptions({ breaks: true })
function renderMd(text) { return marked.parse(text || '') }
function parseSources(s) { try { return JSON.parse(s) } catch { return [] } }
function formatDate(str) { return str ? new Date(str).toLocaleDateString() : '' }

function previewImage(src) {
  previewUrl.value = `/api/v1/docs/${src.doc_id}/file`
  previewFilename.value = src.filename
  previewVisible.value = true
}
function onThumbnailError(e) { e.target.style.display = 'none' }

async function loadSessions() {
  const res = await chatApi.listSessions({ page: 1, page_size: 50 })
  sessions.value = res.data || []
}

async function switchSession(s) {
  currentSession.value = s
  const res = await chatApi.getMessages(s.id)
  messages.value = res.data || []
  scrollToBottom()
}

async function loadKbs() {
  const res = await kbApi.list({ page: 1, page_size: 100 })
  kbList.value = res.data || []
}

async function createSession() {
  const res = await chatApi.createSession(newSessionForm)
  sessions.value.unshift(res.data)
  await switchSession(res.data)
  showNewSession.value = false
}

async function deleteSession(s) {
  await chatApi.deleteSession(s.id)
  sessions.value = sessions.value.filter(x => x.id !== s.id)
  if (currentSession.value?.id === s.id) {
    currentSession.value = null
    messages.value = []
  }
}

function onInputKeydown(e) {
  if (e.shiftKey) return
  if (e.key !== 'Enter' && e.key !== 'NumpadEnter') return
  if (e.isComposing) return
  e.preventDefault()
  sendMessage()
}

async function sendMessage() {
  const text = inputText.value.trim()
  if (!text || streaming.value) return

  // 显示用户消息
  messages.value.push({ _tmpId: Date.now(), role: 'user', content: text })
  inputText.value = ''
  scrollToBottom()

  streaming.value = true
  streamBuffer.value = ''
  let sources = []

  try {
    const token = localStorage.getItem('token')
    const res = await fetch('/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({
        session_id: currentSession.value.id,
        message: text,
        stream: true,
      }),
    })

    const reader = res.body.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      const lines = decoder.decode(value).split('\n')
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const data = JSON.parse(line.slice(6))
        if (data.type === 'chunk') {
          streamBuffer.value += data.content
          scrollToBottom()
        } else if (data.type === 'sources') {
          sources = data.sources
        } else if (data.type === 'done') {
          messages.value.push({
            id: data.message_id,
            role: 'assistant',
            content: streamBuffer.value,
            sources: sources.length ? JSON.stringify(sources) : null,
          })
          streamBuffer.value = ''
        } else if (data.type === 'error') {
          ElMessage.error(data.message)
        }
      }
    }
  } catch (e) {
    ElMessage.error('发送失败：' + e.message)
  } finally {
    streaming.value = false
    scrollToBottom()
  }
}

function scrollToBottom() {
  nextTick(() => {
    if (messagesEl.value) {
      messagesEl.value.scrollTop = messagesEl.value.scrollHeight
    }
  })
}

onMounted(async () => {
  await Promise.all([loadSessions(), loadKbs()])
  // 从知识库页面跳转过来时自动新建会话
  if (route.query.kbId) {
    newSessionForm.kb_id = Number(route.query.kbId)
    // 用知识库名称作为会话标题
    const kb = kbList.value.find(k => k.id === Number(route.query.kbId))
    if (kb) {
      newSessionForm.title = `【${kb.name}】知识库问答`
    }
    showNewSession.value = true
  }
})
</script>

<style scoped>
.chat-page { display: flex; height: 100%; background: #f5f7fa; }

/* 左侧 */
.session-panel { width: 240px; background: #fff; border-right: 1px solid #e4e7ed; display: flex; flex-direction: column; }
.session-header { padding: 12px 16px; font-weight: 600; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f0f0f0; }
.session-list { flex: 1; overflow-y: auto; }
.session-item { padding: 12px 16px; cursor: pointer; border-bottom: 1px solid #f5f5f5; position: relative; transition: background .15s; }
.session-item:hover, .session-item.active { background: #f0f7ff; }
.session-title { font-size: 14px; color: #303133; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.session-meta { font-size: 12px; color: #c0c4cc; margin-top: 4px; }
.del-icon { position: absolute; right: 12px; top: 50%; transform: translateY(-50%); display: none; color: #f56c6c; }
.session-item:hover .del-icon { display: block; }

/* 右侧 */
.chat-area { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.no-session { flex: 1; display: flex; align-items: center; justify-content: center; }
.messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }

.message { max-width: 80%; }
.message.user { align-self: flex-end; }
.message.assistant { align-self: flex-start; }

.msg-content {
  padding: 12px 16px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.6;
}
.message.user .msg-content { background: #409EFF; color: #fff; border-bottom-right-radius: 4px; }
.message.assistant .msg-content { background: #fff; color: #303133; border: 1px solid #e4e7ed; border-bottom-left-radius: 4px; }
.msg-content :deep(p) { margin: 0 0 8px; }
.msg-content :deep(p:last-child) { margin: 0; }
.msg-content :deep(pre) { background: #f5f7fa; padding: 8px; border-radius: 6px; overflow-x: auto; }
.msg-content :deep(code) { font-family: monospace; font-size: 13px; }

.cursor { animation: blink 1s infinite; font-size: 16px; }
@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0; } }

.sources { margin-top: 10px; font-size: 13px; }
.sources-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #909399;
  margin-bottom: 8px;
  font-weight: 500;
}
.source-card {
  border: 1px solid #e4e7ed;
  border-radius: 6px;
  margin-bottom: 8px;
  overflow: hidden;
}
.source-card:last-child { margin-bottom: 0; }
.source-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: #f5f7fa;
  border-bottom: 1px solid #e4e7ed;
  flex-wrap: wrap;
}
.source-filename {
  font-weight: 600;
  color: #303133;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.source-page { color: #909399; font-size: 12px; }
.source-score { margin-left: auto; }
.source-content {
  padding: 10px 12px;
  color: #606266;
  font-size: 13px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 120px;
  overflow-y: auto;
  background: #fff;
}

.source-thumbnail-wrapper {
  margin: 0 12px 10px;
  border-radius: 6px;
  overflow: hidden;
  max-height: 160px;
  display: inline-block;
  border: 1px solid #e4e7ed;
  cursor: pointer;
}
.source-thumbnail {
  max-width: 280px;
  max-height: 160px;
  object-fit: cover;
  display: block;
  transition: opacity .2s;
}
.source-thumbnail:hover { opacity: 0.85; }

.preview-container { text-align: center; }
.preview-image { max-width: 100%; max-height: 70vh; border-radius: 8px; object-fit: contain; }

/* 输入 */
.input-area { padding: 16px; border-top: 1px solid #e4e7ed; background: #fff; }
.input-actions { margin-top: 8px; display: flex; justify-content: flex-end; gap: 8px; align-items: center; }
</style>
