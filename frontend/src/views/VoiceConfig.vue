<template>
  <div class="page">
    <el-alert type="info" show-icon :closable="false" style="margin-bottom:16px">
      <template #title>
        页面配置优先级高于 .env 环境变量。支持 mp3、wav、m4a、aac、ogg、flac、pcm 等常见音频格式。
      </template>
    </el-alert>

    <el-row :gutter="16">
      <el-col :span="12" v-for="item in voices" :key="item.provider">
        <el-card class="voice-card">
          <template #header>
            <div class="card-header">
              <span class="provider-name">{{ providerLabel(item.provider) }}</span>
              <div class="status-tags">
                <el-tag :type="item.is_configured ? 'warning' : 'info'" size="small">
                  {{ item.is_configured ? '已配置' : '未配置' }}
                </el-tag>
                <el-tag v-if="item.is_default" type="primary" size="small">默认</el-tag>
              </div>
            </div>
          </template>

          <el-form :model="formData[item.provider]" label-position="top" size="small">
            <!-- 百度智能云 -->
            <template v-if="item.provider === 'baidu'">
              <el-form-item label="App ID" required>
                <el-input
                  v-model="formData[item.provider].app_id"
                  :placeholder="item.is_configured ? '已配置，留空不修改' : '请输入百度 App ID'"
                  clearable
                />
              </el-form-item>
              <el-form-item label="API Key" required>
                <el-input
                  v-model="formData[item.provider].api_key"
                  :placeholder="item.is_configured ? '已配置，留空不修改' : '请输入百度 API Key'"
                  clearable
                />
              </el-form-item>
              <el-form-item label="Secret Key" required>
                <el-input
                  v-model="formData[item.provider].secret_key"
                  :placeholder="item.is_configured ? '已配置，留空不修改' : '请输入百度 Secret Key'"
                  show-password
                  clearable
                />
              </el-form-item>
              <el-form-item label="识别模型 (dev_pid)" style="margin-bottom:4px">
                <el-select
                  v-model="formData[item.provider].dev_pid"
                  placeholder="选择识别模型"
                  style="width:100%"
                >
                  <el-option label="普通话（简体中文）" value="1537" />
                  <el-option label="英语" value="1737" />
                  <el-option label="粤语" value="1637" />
                  <el-option label="四川话" value="1637" />
                  <el-option label="普通话 远场" value="15371" />
                </el-select>
              </el-form-item>
              <el-form-item label="音频采样率" style="margin-bottom:4px">
                <el-select v-model="formData[item.provider].rate" style="width:100%">
                  <el-option label="16000 Hz（推荐）" value="16000" />
                  <el-option label="8000 Hz" value="8000" />
                </el-select>
              </el-form-item>
            </template>

            <!-- 阿里云 -->
            <template v-else-if="item.provider === 'aliyun'">
              <el-form-item label="AccessKey ID" required>
                <el-input
                  v-model="formData[item.provider].access_key_id"
                  :placeholder="item.is_configured ? '已配置，留空不修改' : '请输入阿里云 AccessKey ID'"
                  clearable
                />
              </el-form-item>
              <el-form-item label="AccessKey Secret" required>
                <el-input
                  v-model="formData[item.provider].access_key_secret"
                  :placeholder="item.is_configured ? '已配置，留空不修改' : '请输入阿里云 AccessKey Secret'"
                  show-password
                  clearable
                />
              </el-form-item>
              <el-form-item label="AppKey" required>
                <el-input
                  v-model="formData[item.provider].app_key"
                  :placeholder="item.is_configured ? '已配置，留空不修改' : '请输入阿里云 AppKey'"
                  clearable
                />
              </el-form-item>
              <el-form-item label="音频采样率" style="margin-bottom:4px">
                <el-select v-model="formData[item.provider].sample_rate" style="width:100%">
                  <el-option label="16000 Hz（推荐）" value="16000" />
                  <el-option label="8000 Hz" value="8000" />
                </el-select>
              </el-form-item>
            </template>

            <el-form-item style="margin-bottom:0">
              <el-checkbox v-model="formData[item.provider].is_default">
                设为默认语音识别 Provider
              </el-checkbox>
            </el-form-item>
          </el-form>

          <div class="card-actions">
            <el-button type="primary" size="small" @click="save(item.provider)">保存</el-button>
            <el-button size="small" @click="reset(item.provider)">清除配置</el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { voiceApi } from '@/api'

const voices = ref([])
const formData = reactive({})

const PROVIDER_LABELS = {
  baidu: '百度智能云 ASR',
  aliyun: '阿里云语音识别',
}

function providerLabel(p) { return PROVIDER_LABELS[p] || p }

function _defaultForm() {
  return {
    api_key: '',
    secret_key: '',
    app_id: '',
    access_key_id: '',
    access_key_secret: '',
    app_key: '',
    dev_pid: '1537',
    rate: '16000',
    sample_rate: '16000',
    is_default: false,
  }
}

function _parseExtra(extra_params) {
  if (!extra_params) return {}
  try {
    return JSON.parse(extra_params)
  } catch {
    return {}
  }
}

async function loadVoices() {
  const res = await voiceApi.list()
  voices.value = res.data || []
  for (const m of voices.value) {
    const extra = _parseExtra(m.extra_params)
    formData[m.provider] = {
      api_key: '',
      secret_key: '',
      app_id: extra.app_id || '',
      access_key_id: '',
      access_key_secret: '',
      app_key: extra.app_key || '',
      dev_pid: extra.dev_pid || '1537',
      rate: extra.rate || '16000',
      sample_rate: extra.sample_rate || '16000',
      is_default: m.is_default,
    }
  }
}

async function save(provider) {
  const fd = formData[provider]
  const cfg = voices.value.find(v => v.provider === provider)
  const isUpdate = cfg && cfg.is_configured

  // 参数校验：已保存过的配置跳过凭证必填校验
  const missing = []
  if (provider === 'baidu') {
    if (!isUpdate && !fd.api_key) missing.push('API Key')
    if (!fd.app_id) missing.push('App ID')
    if (!isUpdate && !fd.secret_key) missing.push('Secret Key')
  } else if (provider === 'aliyun') {
    if (!isUpdate && !fd.access_key_id) missing.push('AccessKey ID')
    if (!isUpdate && !fd.access_key_secret) missing.push('AccessKey Secret')
    if (!fd.app_key) missing.push('AppKey')
  }
  if (missing.length) {
    ElMessage.error(`请填写完整：${missing.join('、')} 不能为空`)
    return
  }

  const payload = {}
  payload.is_default = fd.is_default

  const extra = {}
  if (provider === 'baidu') {
    extra.app_id = fd.app_id
    if (!isUpdate) {
      payload.api_key = fd.api_key
      payload.api_secret = fd.secret_key
    }
    extra.dev_pid = parseInt(fd.dev_pid)
    extra.rate = parseInt(fd.rate)
  } else if (provider === 'aliyun') {
    if (!isUpdate) {
      payload.api_key = fd.access_key_id
      payload.api_secret = fd.access_key_secret
    }
    extra.app_key = fd.app_key
    extra.sample_rate = parseInt(fd.sample_rate)
  }
  payload.extra_params = JSON.stringify(extra)

  try {
    await voiceApi.upsert(provider, payload)
    ElMessage.success('配置已保存')
    // 保存后清空敏感字段
    if (provider === 'baidu') {
      formData[provider].api_key = ''
      formData[provider].app_id = ''
      formData[provider].secret_key = ''
    } else {
      formData[provider].access_key_id = ''
      formData[provider].access_key_secret = ''
      formData[provider].app_key = ''
    }
    await loadVoices()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '保存失败')
  }
}

async function reset(provider) {
  await ElMessageBox.confirm(
    `确定清除 ${providerLabel(provider)} 的配置？`,
    '提示',
    { type: 'warning' }
  )
  try {
    await voiceApi.remove(provider)
    ElMessage.success('配置已清除')
    await loadVoices()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '操作失败')
  }
}

onMounted(loadVoices)
</script>

<style scoped>
.page { padding: 0; }
.voice-card { margin-bottom: 16px; }
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.provider-name { font-weight: 600; font-size: 15px; }
.status-tags { display: flex; gap: 6px; flex-wrap: wrap; }
.card-actions {
  display: flex;
  gap: 8px;
  margin-top: 4px;
  flex-wrap: wrap;
}
</style>
