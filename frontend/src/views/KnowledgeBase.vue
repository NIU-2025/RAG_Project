<template>
  <div class="kb-page">
    <div class="toolbar">
      <el-button type="primary" :icon="Plus" @click="handleCreateKb">新建知识库</el-button>
    </div>

    <el-row :gutter="16" class="kb-grid">
      <el-col v-for="kb in kbList" :key="kb.id" :span="6">
        <el-card class="kb-card" shadow="hover">
          <div class="kb-header">
            <el-icon class="kb-icon"><Folder /></el-icon>
          </div>
          <div class="kb-name">{{ kb.name }}</div>
          <div class="kb-desc">{{ kb.description || '暂无描述' }}</div>
          <div class="kb-meta">
            <span>{{ kb.doc_count }} 个文档</span>
            <span>{{ formatDate(kb.updated_at) }}</span>
          </div>
          <div class="kb-actions">
            <el-button size="small" @click="goDocs(kb.id)">管理文档</el-button>
            <el-button size="small" type="primary" @click="goChat(kb.id)">开始问答</el-button>
            <el-dropdown @command="(cmd) => handleKbAction(cmd, kb)">
              <el-button size="small" :icon="MoreFilled" circle />
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item command="edit">编辑</el-dropdown-item>
                  <el-dropdown-item command="delete" divided style="color:#f56c6c">删除</el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-empty v-if="!loading && kbList.length === 0" description="还没有知识库，点击右上角新建" />

    <!-- 新建/编辑弹窗 -->
    <el-dialog v-model="showCreate" :title="editingKb ? '编辑知识库' : '新建知识库'" width="480px" @close="handleDialogClose">
      <el-form :model="kbForm" label-position="top">
        <el-form-item label="名称" required>
          <el-input v-model="kbForm.name" placeholder="知识库名称" />
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="kbForm.description" type="textarea" :rows="3" placeholder="描述（可选）" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showCreate = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="saveKb">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, MoreFilled } from '@element-plus/icons-vue'
import { kbApi } from '@/api'

const router = useRouter()
const kbList = ref([])
const loading = ref(false)
const showCreate = ref(false)
const saving = ref(false)
const editingKb = ref(null)
const kbForm = reactive({ name: '', description: '' })

function handleCreateKb() {
  editingKb.value = null
  Object.assign(kbForm, { name: '', description: '' })
  showCreate.value = true
}

async function loadKbs() {
  loading.value = true
  try {
    const res = await kbApi.list({ page: 1, page_size: 100 })
    kbList.value = res.data || []
  } finally {
    loading.value = false
  }
}

async function saveKb() {
  if (!kbForm.name.trim()) { ElMessage.warning('请输入知识库名称'); return }
  saving.value = true
  try {
    if (editingKb.value) {
      await kbApi.update(editingKb.value.id, kbForm)
      ElMessage.success('已更新')
    } else {
      await kbApi.create(kbForm)
      ElMessage.success('创建成功')
    }
    showCreate.value = false
    loadKbs()
  } finally {
    saving.value = false
  }
}

function handleKbAction(cmd, kb) {
  if (cmd === 'edit') {
    editingKb.value = kb
    Object.assign(kbForm, { name: kb.name, description: kb.description })
    showCreate.value = true
  } else if (cmd === 'delete') {
    ElMessageBox.confirm(`确定删除知识库「${kb.name}」及其所有文档？`, '警告', { type: 'warning' })
      .then(async () => {
        await kbApi.delete(kb.id)
        ElMessage.success('已删除')
        loadKbs()
      })
  }
}

function handleDialogClose() {
  editingKb.value = null
  Object.assign(kbForm, { name: '', description: '' })
}

function goDocs(kbId) { router.push(`/kb/${kbId}/docs`) }
function goChat(kbId) { router.push({ path: '/chat', query: { kbId } }) }
function formatDate(str) { return str ? new Date(str).toLocaleDateString() : '' }

onMounted(loadKbs)
</script>

<style scoped>
.kb-page { padding: 20px; }
.toolbar { margin-bottom: 20px; }
.kb-grid { }
.kb-card { cursor: default; margin-bottom: 16px; }
.kb-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.kb-icon { font-size: 32px; color: #409EFF; }
.kb-name { font-size: 16px; font-weight: 600; margin-bottom: 6px; color: #303133; }
.kb-desc { font-size: 13px; color: #909399; margin-bottom: 12px; height: 38px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
.kb-meta { display: flex; justify-content: space-between; font-size: 12px; color: #c0c4cc; margin-bottom: 12px; }
.kb-actions { display: flex; gap: 6px; }
</style>
