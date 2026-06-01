<template>
  <div class="user-manage">
    <div class="toolbar">
      <el-input v-model="keyword" placeholder="搜索用户名/邮箱" style="width:260px" clearable @keyup.enter="loadUsers" />
      <el-button type="primary" @click="loadUsers">搜索</el-button>
    </div>

    <el-table :data="users" v-loading="loading" stripe>
      <el-table-column prop="id" label="ID" width="60" />
      <el-table-column prop="username" label="用户名" />
      <el-table-column prop="email" label="邮箱" />
      <el-table-column prop="is_active" label="状态" width="80">
        <template #default="{ row }">
          <el-tag :type="row.is_active ? 'success' : 'danger'" size="small">
            {{ row.is_active ? '正常' : '禁用' }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="roles" label="角色">
        <template #default="{ row }">
          <el-tag v-for="r in row.roles" :key="r.id" size="small" class="mr-4">{{ r.name }}</el-tag>
          <span v-if="!row.roles.length" style="color:#909399;font-size:12px">未分配角色</span>
        </template>
      </el-table-column>
      <el-table-column prop="created_at" label="注册时间" width="170">
        <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
      </el-table-column>
      <el-table-column label="操作" width="160" fixed="right">
        <template #default="{ row }">
          <el-button type="primary" link size="small" @click="openAssignRoles(row)">分配角色</el-button>
          <el-button type="danger" link size="small" @click="handleDelete(row)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-pagination
      v-model:current-page="page"
      :page-size="20"
      :total="total"
      layout="total, prev, pager, next"
      @current-change="loadUsers"
      style="margin-top:16px"
    />

    <!-- 分配角色对话框 -->
    <el-dialog v-model="roleDialogVisible" title="分配角色" width="500px">
      <el-form label-width="80">
        <el-form-item label="用户">{{ currentUser?.username }}</el-form-item>
        <el-form-item label="选择角色">
          <el-checkbox-group v-model="selectedRoleIds">
            <el-checkbox v-for="r in allRoles" :key="r.id" :value="r.id" :disabled="r.is_admin === true">
              {{ r.name }}
              <span v-if="r.is_admin === true" style="color:#909399;font-size:12px">（系统角色不可分配）</span>
            </el-checkbox>
          </el-checkbox-group>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="roleDialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="handleAssignRoles">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { userApi } from '@/api'

const loading = ref(false)
const users = ref([])
const allRoles = ref([])
const total = ref(0)
const page = ref(1)
const keyword = ref('')

const roleDialogVisible = ref(false)
const currentUser = ref(null)
const selectedRoleIds = ref([])
const saving = ref(false)

async function loadUsers() {
  loading.value = true
  try {
    const res = await userApi.list({ page: page.value, page_size: 20, keyword: keyword.value || undefined })
    users.value = res.data || []
    total.value = res.total || 0
  } catch (e) {
    ElMessage.error('加载用户列表失败')
  } finally {
    loading.value = false
  }
}

async function loadAllRoles() {
  try {
    const res = await userApi.getAllRoles()
    allRoles.value = res.data || []
  } catch (e) {
    ElMessage.error('加载角色列表失败')
  }
}

function openAssignRoles(user) {
  currentUser.value = user
  selectedRoleIds.value = user.roles.map(r => r.id)
  roleDialogVisible.value = true
}

async function handleAssignRoles() {
  saving.value = true
  try {
    await userApi.assignRoles(currentUser.value.id, { role_ids: selectedRoleIds.value })
    ElMessage.success('角色分配成功')
    roleDialogVisible.value = false
    await loadUsers()
  } catch (e) {
    ElMessage.error(e.detail || '分配失败')
  } finally {
    saving.value = false
  }
}

async function handleDelete(user) {
  await ElMessageBox.confirm(`确定删除用户「${user.username}」吗？`, '提示', { type: 'warning' })
  try {
    await userApi.delete(user.id)
    ElMessage.success('删除成功')
    await loadUsers()
  } catch (e) {
    ElMessage.error(e.detail || '删除失败')
  }
}

function formatTime(ts) {
  if (!ts) return '-'
  return new Date(ts).toLocaleString('zh-CN')
}

onMounted(() => {
  loadUsers()
  loadAllRoles()
})
</script>

<style scoped>
.user-manage { padding: 20px; }
.toolbar { display: flex; gap: 12px; margin-bottom: 16px; align-items: center; }
.mr-4 { margin-right: 4px; }
</style>
