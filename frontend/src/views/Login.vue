<template>
  <div class="login-page">
    <el-card class="login-card">
      <template #header>
        <h2 class="card-title">RAG 知识库问答系统</h2>
      </template>

      <el-tabs v-model="activeTab">
        <el-tab-pane label="登录" name="login">
          <el-form :model="loginForm" @submit.prevent="handleLogin" label-position="top">
            <el-form-item label="用户名">
              <el-input v-model="loginForm.username" placeholder="请输入用户名" clearable />
            </el-form-item>
            <el-form-item label="密码">
              <el-input v-model="loginForm.password" type="password" placeholder="请输入密码" show-password @keyup.enter="handleLogin" />
            </el-form-item>
            <el-button type="primary" :loading="loading" @click="handleLogin" class="submit-btn">登录</el-button>
          </el-form>
        </el-tab-pane>

        <el-tab-pane label="注册" name="register">
          <el-form :model="registerForm" @submit.prevent="handleRegister" label-position="top">
            <el-form-item label="用户名">
              <el-input v-model="registerForm.username" placeholder="3-64个字符" clearable />
            </el-form-item>
            <el-form-item label="邮箱">
              <el-input v-model="registerForm.email" placeholder="请输入邮箱" clearable />
            </el-form-item>
            <el-form-item label="密码">
              <el-input v-model="registerForm.password" type="password" placeholder="至少6个字符" show-password />
            </el-form-item>
            <el-button type="primary" :loading="loading" @click="handleRegister" class="submit-btn">注册</el-button>
          </el-form>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useAuthStore, findFirstAccessibleRoute } from '@/stores/auth'
import { authApi } from '@/api'

const router = useRouter()
const auth = useAuthStore()
const activeTab = ref('login')
const loading = ref(false)

const loginForm = reactive({ username: '', password: '' })
const registerForm = reactive({ username: '', email: '', password: '' })

async function handleLogin() {
  if (!loginForm.username || !loginForm.password) {
    ElMessage.warning('请填写完整信息')
    return
  }
  loading.value = true
  try {
    await auth.login(loginForm.username, loginForm.password)
    router.push(findFirstAccessibleRoute(auth.menuPermissionCodes, auth.isAdmin))
    ElMessage.success('登录成功')
  } finally {
    loading.value = false
  }
}

async function handleRegister() {
  if (!registerForm.username || !registerForm.email || !registerForm.password) {
    ElMessage.warning('请填写完整信息')
    return
  }
  loading.value = true
  try {
    await authApi.register(registerForm)
    ElMessage.success('注册成功，请登录')
    activeTab.value = 'login'
    loginForm.username = registerForm.username
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #F5F5F5;
  margin: 0;
  padding: 0;
  overflow: hidden;
}
.login-card {
  width: 420px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}
.card-title {
  text-align: center;
  margin: 0;
  font-size: 20px;
  color: #303133;
}
.submit-btn {
  width: 100%;
  margin-top: 8px;
}
</style>
