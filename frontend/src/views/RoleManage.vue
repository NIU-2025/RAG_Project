<template>
  <div class="role-manage">
    <div class="toolbar">
      <el-button type="primary" @click="openCreate">新建角色</el-button>
    </div>

    <el-table :data="roles" v-loading="loading" stripe>
      <el-table-column prop="id" label="ID" width="60" />
      <el-table-column prop="name" label="角色名称">
        <template #default="{ row }">
          <span v-if="row.is_admin" style="color:#409EFF;font-weight:600">{{ row.name }}</span>
          <span v-else>{{ row.name }}</span>
          <el-tag v-if="row.is_admin" type="danger" size="small" style="margin-left:6px">超级管理员</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="description" label="描述" />
      <el-table-column prop="permissions" label="菜单权限">
        <template #default="{ row }">
          <span v-if="row.is_admin" style="color:#909399">拥有所有权限</span>
          <el-tag v-else v-for="p in row.permissions.filter(x=>x.type==='menu')" :key="p.id" size="small" class="mr-4">{{ p.name }}</el-tag>
          <span v-if="!row.is_admin && !row.permissions.filter(x=>x.type==='menu').length" style="color:#909399;font-size:12px">未分配</span>
        </template>
      </el-table-column>
      <el-table-column prop="kb_permission_ids" label="知识库权限">
        <template #default="{ row }">
          <span v-if="row.is_admin" style="color:#909399">拥有所有知识库权限</span>
          <span v-else-if="!row.kb_permission_ids?.length" style="color:#909399;font-size:12px">未分配</span>
          <el-tag v-else v-for="kid in row.kb_permission_ids" :key="kid" size="small" type="warning" class="mr-4">{{ kbNameMap[kid] || `KB #${kid}` }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="280" fixed="right">
        <template #default="{ row }">
          <el-button type="primary" link size="small" @click="openEdit(row)">编辑</el-button>
          <el-button type="success" link size="small" @click="openMenuPerm(row)">配置权限</el-button>
          <el-button v-if="!row.is_admin" type="warning" link size="small" @click="openKbPerm(row)">知识库权限</el-button>
          <el-button v-if="!row.is_admin" type="danger" link size="small" @click="handleDelete(row)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- 新建/编辑角色对话框 -->
    <el-dialog v-model="formVisible" :title="editingId ? '编辑角色' : '新建角色'" width="460px">
      <el-form :model="form" label-width="80">
        <el-form-item label="角色名称" required>
          <el-input v-model="form.name" placeholder="请输入角色名称" maxlength="64" />
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="form.description" type="textarea" :rows="3" placeholder="可选" maxlength="255" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="formVisible = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="handleSave">保存</el-button>
      </template>
    </el-dialog>

    <!-- 配置菜单权限对话框 -->
    <el-dialog v-model="menuPermVisible" title="配置菜单权限" width="560px">
      <div v-if="currentRole">
        <p style="margin-bottom:16px">
          角色：<strong>{{ currentRole.name }}</strong>
        </p>
        <div class="perm-section">
          <div class="perm-section-title">菜单权限</div>
          <el-checkbox-group v-model="selectedMenuPermIds">
            <el-checkbox
              v-for="p in menuPermissions"
              :key="p.id"
              :value="p.id"
              style="width:180px;margin-bottom:8px"
            >{{ p.name }}</el-checkbox>
          </el-checkbox-group>
        </div>
      </div>
      <template #footer>
        <el-button @click="menuPermVisible = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="handleSaveMenuPerms">保存权限</el-button>
      </template>
    </el-dialog>

    <!-- 知识库权限对话框 -->
    <el-dialog v-model="kbPermVisible" title="知识库权限" width="560px">
      <div v-if="currentRole">
        <p style="margin-bottom:16px">
          角色：<strong>{{ currentRole.name }}</strong>
          <span style="color:#909399;font-size:12px;margin-left:8px">勾选后该角色可访问对应知识库</span>
        </p>
        <div class="perm-section">
          <el-checkbox-group v-model="selectedKbIds">
            <el-checkbox
              v-for="kb in allKbs"
              :key="kb.id"
              :value="kb.id"
              style="width:200px;margin-bottom:8px"
            >{{ kb.name }}</el-checkbox>
          </el-checkbox-group>
          <div v-if="!allKbs.length" style="color:#909399;font-size:13px">暂无可分配的知识库</div>
        </div>
      </div>
      <template #footer>
        <el-button @click="kbPermVisible = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="handleSaveKbPerms">保存权限</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { roleApi } from '@/api'

const loading = ref(false)
const roles = ref([])
const menuPermissions = ref([])
const allKbs = ref([])

const formVisible = ref(false)
const form = ref({ name: '', description: '' })
const editingId = ref(null)
const saving = ref(false)

const menuPermVisible = ref(false)
const currentRole = ref(null)
const selectedMenuPermIds = ref([])

const kbPermVisible = ref(false)
const selectedKbIds = ref([])

const kbNameMap = computed(() => {
  const m = {}
  for (const kb of allKbs.value) m[kb.id] = kb.name
  return m
})

async function loadRoles() {
  loading.value = true
  try {
    const res = await roleApi.list({ page: 1, page_size: 100 })
    roles.value = res.data || []
  } catch (e) {
    ElMessage.error('加载角色列表失败')
  } finally {
    loading.value = false
  }
}

async function loadPermOptions() {
  try {
    const res = await roleApi.getAllPermissions()
    menuPermissions.value = res.data?.menu_permissions || []
    allKbs.value = (res.data?.kb_permissions || []).map(k => ({ id: k.kb_id, name: k.kb_name }))
  } catch (e) {
    ElMessage.error('加载权限选项失败')
  }
}

function openCreate() {
  editingId.value = null
  form.value = { name: '', description: '' }
  formVisible.value = true
}

async function openEdit(role) {
  try {
    const res = await roleApi.get(role.id)
    const data = res?.data || res
    editingId.value = role.id
    form.value = { name: data.name, description: data.description || '' }
    formVisible.value = true
  } catch (e) {
    ElMessage.error('加载角色详情失败')
  }
}

async function handleSave() {
  if (!form.value.name?.trim()) {
    ElMessage.warning('请填写角色名称')
    return
  }
  saving.value = true
  try {
    if (editingId.value) {
      await roleApi.update(editingId.value, form.value)
      ElMessage.success('更新成功')
    } else {
      await roleApi.create(form.value)
      ElMessage.success('创建成功')
    }
    formVisible.value = false
    await loadRoles()
  } catch (e) {
    ElMessage.error(e.detail || '操作失败')
  } finally {
    saving.value = false
  }
}

async function handleDelete(role) {
  await ElMessageBox.confirm(`确定删除角色「${role.name}」吗？`, '提示', { type: 'warning' })
  try {
    await roleApi.delete(role.id)
    ElMessage.success('删除成功')
    await loadRoles()
  } catch (e) {
    ElMessage.error(e.detail || '删除失败')
  }
}

// 菜单权限
async function openMenuPerm(role) {
  try {
    const res = await roleApi.get(role.id)
    const data = res?.data || res
    currentRole.value = data
    selectedMenuPermIds.value = (data.permissions || [])
      .filter(p => p.type === 'menu')
      .map(p => p.id)
    menuPermVisible.value = true
  } catch (e) {
    ElMessage.error('加载角色权限失败')
  }
}

async function handleSaveMenuPerms() {
  saving.value = true
  try {
    await roleApi.updateMenuPermissions(currentRole.value.id, selectedMenuPermIds.value)
    ElMessage.success('菜单权限配置成功')
    menuPermVisible.value = false
    await loadRoles()
  } catch (e) {
    ElMessage.error(e.detail || '保存失败')
  } finally {
    saving.value = false
  }
}

// 知识库权限
async function openKbPerm(role) {
  try {
    const res = await roleApi.get(role.id)
    const data = res?.data || res
    currentRole.value = data
    selectedKbIds.value = data.kb_permission_ids || []
    kbPermVisible.value = true
  } catch (e) {
    ElMessage.error('加载知识库权限失败')
  }
}

async function handleSaveKbPerms() {
  saving.value = true
  try {
    await roleApi.updateKbPermissions(currentRole.value.id, selectedKbIds.value)
    ElMessage.success('知识库权限配置成功')
    kbPermVisible.value = false
    await loadRoles()
  } catch (e) {
    ElMessage.error(e.detail || '保存失败')
  } finally {
    saving.value = false
  }
}

onMounted(() => {
  loadRoles()
  loadPermOptions()
})
</script>

<style scoped>
.role-manage { padding: 20px; }
.toolbar { display: flex; gap: 12px; margin-bottom: 16px; }
.mr-4 { margin-right: 4px; }
.perm-section {
  margin-bottom: 20px;
  padding: 12px 16px;
  background: #f5f7fa;
  border-radius: 8px;
}
.perm-section-title {
  font-weight: 600;
  color: #303133;
  margin-bottom: 12px;
  font-size: 14px;
}
</style>
