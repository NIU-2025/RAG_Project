<template>
  <el-container class="layout">
    <el-aside width="200px" class="aside">
      <div class="logo">RAG 知识库</div>
      <el-menu :router="true" :default-active="route.path" background-color="#1a1a2e" text-color="#ccc" active-text-color="#fff">
        <el-menu-item v-if="auth.isAdmin || auth.hasPermission('stats')" index="/dashboard">
          <el-icon><DataAnalysis /></el-icon>
          <span>仪表盘统计</span>
        </el-menu-item>
        <el-menu-item v-if="auth.isAdmin || auth.hasPermission('kb_manage')" index="/kb">
          <el-icon><Folder /></el-icon>
          <span>知识库</span>
        </el-menu-item>
        <el-menu-item v-if="auth.isAdmin || auth.hasPermission('chat')" index="/chat">
          <el-icon><ChatDotRound /></el-icon>
          <span>问答对话</span>
        </el-menu-item>
        <el-menu-item v-if="auth.isAdmin || auth.hasPermission('model_config')" index="/models">
          <el-icon><Setting /></el-icon>
          <span>模型管理</span>
        </el-menu-item>
        <el-menu-item v-if="auth.isAdmin || auth.hasPermission('voice_config')" index="/voice">
          <el-icon><Microphone /></el-icon>
          <span>语音配置</span>
        </el-menu-item>
        <el-menu-item v-if="auth.isAdmin || auth.hasPermission('user_manage')" index="/users">
          <el-icon><User /></el-icon>
          <span>用户管理</span>
        </el-menu-item>
        <el-menu-item v-if="auth.isAdmin || auth.hasPermission('role_manage')" index="/roles">
          <el-icon><Key /></el-icon>
          <span>角色管理</span>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <el-container>
      <el-header class="header">
        <span class="header-title">{{ pageTitle }}</span>
        <div class="header-right">
          <el-tag v-if="auth.isAdmin" type="danger" size="small">超级管理员</el-tag>
          <el-tag v-else-if="auth.roles.length" type="success" size="small">{{ auth.roles.map(r => r.name).join(', ') }}</el-tag>
          <span class="username">{{ auth.user?.username }}</span>
          <el-button link @click="handleLogout">退出</el-button>
        </div>
      </el-header>
      <el-main class="main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { ElMessageBox } from 'element-plus'
import { Folder, ChatDotRound, Setting, Microphone, DataAnalysis, User, Key } from '@element-plus/icons-vue'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()

const pageTitle = computed(() => {
  const map = {
    '/kb': '知识库管理',
    '/chat': '问答对话',
    '/dashboard': '仪表盘统计',
    '/models': '模型管理',
    '/voice': '语音配置',
    '/users': '用户管理',
    '/roles': '角色管理',
  }
  return map[route.path] || ''
})

async function handleLogout() {
  await ElMessageBox.confirm('确定退出登录？', '提示', { type: 'warning' })
  auth.logout()
  router.push('/login')
}
</script>

<style scoped>
.layout { height: 100vh; }
.aside {
  background: #1a1a2e;
  display: flex;
  flex-direction: column;
}
.logo {
  color: #fff;
  font-size: 18px;
  font-weight: 700;
  padding: 20px 16px;
  border-bottom: 1px solid #333;
}
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #eee;
  background: #fff;
}
.header-title { font-size: 16px; font-weight: 600; color: #303133; }
.header-right { display: flex; align-items: center; gap: 10px; }
.username { color: #606266; font-size: 14px; }
.main { background: #f5f7fa; overflow-y: auto; }
</style>
