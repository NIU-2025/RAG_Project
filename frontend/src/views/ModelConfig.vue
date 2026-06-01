<template>
  <div class="page">
    <el-alert type="info" show-icon :closable="false" style="margin-bottom:16px">
      <template #title>
        页面配置优先级高于 .env 环境变量。<b>保存配置后请点击"测试连接"验证是否可用。</b>
      </template>
    </el-alert>

    <el-row :gutter="16">
      <el-col :span="12" v-for="item in models" :key="item.provider">
        <el-card class="model-card">
          <template #header>
            <div class="card-header">
              <span class="provider-name">{{ providerLabel(item.provider) }}</span>
              <div class="status-tags">
                <!-- 凭证是否填写 -->
                <el-tag :type="item.is_configured ? 'warning' : 'info'" size="small">
                  {{ item.is_configured ? '已配置' : '未配置' }}
                </el-tag>
                <!-- 连通性测试结果 -->
                <el-tag
                  v-if="testResults[item.provider] !== undefined"
                  :type="testResults[item.provider] ? 'success' : 'danger'"
                  size="small"
                >
                  {{ testResults[item.provider] ? '可用' : '不可用' }}
                </el-tag>
                <el-tag v-else type="info" size="small">未测试</el-tag>
              </div>
            </div>
          </template>

          <!-- 测试结果消息 -->
          <el-alert
            v-if="testMessages[item.provider]"
            :type="testResults[item.provider] ? 'success' : 'error'"
            :description="testMessages[item.provider]"
            show-icon
            :closable="false"
            style="margin-bottom:12px;font-size:12px"
          />

          <el-form :model="formData[item.provider]" label-position="top" size="small">
            <!-- API Key -->
            <el-form-item v-if="!['ollama', 'lmstudio'].includes(item.provider)" label="API Key">
              <el-input
                v-model="formData[item.provider].api_key"
                :placeholder="item.api_key ? '已配置（****），留空不修改' : '请输入 API Key'"
                show-password
                clearable
              />
            </el-form-item>

            <!-- qianfan secret key -->
            <el-form-item v-if="item.provider === 'qianfan'" label="Secret Key">
              <el-input
                v-model="formData[item.provider].api_secret"
                :placeholder="item.api_secret ? '已配置（****），留空不修改' : '请输入 Secret Key'"
                show-password
                clearable
              />
            </el-form-item>

            <!-- Base URL -->
            <el-form-item v-if="['openai', 'deepseek', 'ollama', 'lmstudio'].includes(item.provider)" label="Base URL">
              <el-input
                v-model="formData[item.provider].base_url"
                :placeholder="defaultBaseUrl(item.provider)"
                clearable
              />
            </el-form-item>

            <!-- Model Name -->
            <el-form-item label="默认模型">
              <el-input
                v-model="formData[item.provider].model_name"
                :placeholder="defaultModel(item.provider)"
                clearable
              />
            </el-form-item>
          </el-form>

          <div class="card-actions">
            <el-button type="primary" size="small" @click="save(item.provider)">保存</el-button>
            <el-button
              type="success"
              size="small"
              :loading="testingMap[item.provider]"
              :disabled="!item.is_configured"
              @click="testConnection(item.provider)"
            >
              测试连接
            </el-button>
            <el-button size="small" @click="reset(item.provider)">重置为环境变量</el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { modelsApi } from '@/api'

const models = ref([])
const formData = reactive({})
const testResults = reactive({})   // provider -> true/false
const testMessages = reactive({})  // provider -> string
const testingMap = reactive({})    // provider -> loading bool

const PROVIDER_LABELS = {
  openai: 'OpenAI',
  deepseek: 'DeepSeek',
  dashscope: '通义千问 (DashScope)',
  qianfan: '文心一言 (千帆)',
  ollama: 'Ollama (本地)',
  lmstudio: 'LM Studio (本地)',
}

const DEFAULT_BASE_URLS = {
  openai: 'https://api.openai.com/v1',
  deepseek: 'https://api.deepseek.com/v1',
  ollama: 'http://localhost:11434',
  lmstudio: 'http://localhost:1234/v1',
}

const DEFAULT_MODELS = {
  openai: 'gpt-4o-mini',
  deepseek: 'deepseek-chat',
  dashscope: 'qwen-max',
  qianfan: 'ERNIE-4.0-8K',
  ollama: 'qwen2.5:7b',
  lmstudio: '',
}

function providerLabel(p) { return PROVIDER_LABELS[p] || p }
function defaultBaseUrl(p) { return DEFAULT_BASE_URLS[p] || '' }
function defaultModel(p) { return DEFAULT_MODELS[p] || '' }

async function loadModels() {
  const res = await modelsApi.list()
  models.value = res.data || []
  for (const m of models.value) {
    if (!formData[m.provider]) {
      formData[m.provider] = { api_key: '', api_secret: '', base_url: m.base_url || '', model_name: m.model_name || '' }
    }
  }
}

async function save(provider) {
  const data = formData[provider]
  const payload = {}
  if (data.api_key) payload.api_key = data.api_key
  if (data.api_secret) payload.api_secret = data.api_secret
  if (data.base_url !== undefined) payload.base_url = data.base_url
  if (data.model_name !== undefined) payload.model_name = data.model_name

  try {
    await modelsApi.upsert(provider, payload)
    ElMessage.success('配置已保存，请点击"测试连接"验证')
    formData[provider].api_key = ''
    formData[provider].api_secret = ''
    // 保存后清除旧测试结果
    delete testResults[provider]
    delete testMessages[provider]
    await loadModels()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '保存失败')
  }
}

async function testConnection(provider) {
  testingMap[provider] = true
  try {
    const res = await modelsApi.test(provider)
    testResults[provider] = res.data?.is_available ?? false
    testMessages[provider] = res.data?.message || ''
  } catch (e) {
    testResults[provider] = false
    testMessages[provider] = e.response?.data?.detail || '测试请求失败'
  } finally {
    testingMap[provider] = false
  }
}

async function reset(provider) {
  await ElMessageBox.confirm(`确定重置 ${providerLabel(provider)} 的配置（将使用环境变量）？`, '提示', { type: 'warning' })
  try {
    await modelsApi.remove(provider)
    ElMessage.success('已重置为环境变量配置')
    delete testResults[provider]
    delete testMessages[provider]
    await loadModels()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '操作失败')
  }
}

onMounted(loadModels)
</script>

<style scoped>
.page { padding: 0; }
.model-card { margin-bottom: 16px; }
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.provider-name { font-weight: 600; font-size: 15px; }
.status-tags { display: flex; gap: 6px; }
.card-actions {
  display: flex;
  gap: 8px;
  margin-top: 4px;
  flex-wrap: wrap;
}
</style>
