<template>
  <div class="welcome-page">
    <div class="welcome-card">
      <div class="welcome-icon">
        <svg viewBox="0 0 64 64" width="80" height="80" fill="none">
          <circle cx="32" cy="32" r="30" stroke="#409EFF" stroke-width="3"/>
          <path d="M20 32 L28 40 L44 24" stroke="#409EFF" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <h1 class="welcome-title">欢迎访问 RAG 智能问答系统</h1>
      <p class="welcome-subtitle">您当前尚未分配任何角色，暂时无法使用系统功能</p>
      <p class="welcome-tip">请联系管理员为您分配角色后，再重新登录使用</p>
      <div class="welcome-info">
        <div class="info-item">
          <span class="info-label">当前账号</span>
          <span class="info-value">{{ auth.user?.username }}</span>
        </div>
        <div class="info-item">
          <span class="info-label">当前邮箱</span>
          <span class="info-value">{{ auth.user?.email }}</span>
        </div>
      </div>
      <div class="welcome-actions">
        <el-button type="primary" @click="handleLogout">退出登录</el-button>
        <el-button @click="handleRefresh" :loading="refreshing">刷新权限</el-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const auth = useAuthStore()
const refreshing = ref(false)

async function handleRefresh() {
  refreshing.value = true
  try {
    await auth.fetchPermissions()
    if (!auth.hasNoRole()) {
      ElMessage.success('权限已更新，即将跳转...')
      router.push('/')
    } else {
      ElMessage.warning('仍未分配角色，请联系管理员')
    }
  } finally {
    refreshing.value = false
  }
}

async function handleLogout() {
  await ElMessageBox.confirm('确定退出登录？', '提示', { type: 'warning' })
  auth.logout()
  router.push('/login')
}
</script>

<style scoped>
.welcome-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 50%, #a5d6a7 100%);
  margin: 0;
  padding: 20px;
}
.welcome-card {
  background: #fff;
  border-radius: 16px;
  padding: 48px 40px;
  max-width: 480px;
  width: 100%;
  text-align: center;
  box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}
.welcome-icon {
  margin-bottom: 24px;
}
.welcome-title {
  font-size: 24px;
  font-weight: 700;
  color: #303133;
  margin: 0 0 12px;
}
.welcome-subtitle {
  font-size: 15px;
  color: #606266;
  margin: 0 0 8px;
}
.welcome-tip {
  font-size: 14px;
  color: #909399;
  margin: 0 0 32px;
}
.welcome-info {
  background: #f5f7fa;
  border-radius: 8px;
  padding: 16px 20px;
  margin-bottom: 28px;
  text-align: left;
}
.info-item {
  display: flex;
  justify-content: space-between;
  padding: 6px 0;
  font-size: 14px;
}
.info-item + .info-item {
  border-top: 1px solid #ebeef5;
}
.info-label {
  color: #909399;
}
.info-value {
  color: #303133;
  font-weight: 500;
}
.welcome-actions {
  display: flex;
  gap: 12px;
  justify-content: center;
}
</style>
