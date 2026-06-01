import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/Login.vue'),
    meta: { public: true }
  },
  {
    path: '/',
    component: () => import('@/views/Layout.vue'),
    children: [
      { path: '', redirect: '/dashboard' },
      { path: 'kb', name: 'KnowledgeBase', component: () => import('@/views/KnowledgeBase.vue'), meta: { perm: 'kb_manage' } },
      { path: 'kb/:kbId/docs', name: 'Documents', component: () => import('@/views/Documents.vue'), meta: { perm: 'kb_manage' } },
      { path: 'chat', name: 'Chat', component: () => import('@/views/Chat.vue'), meta: { perm: 'chat' } },
      {
        path: 'dashboard',
        name: 'Dashboard',
        component: () => import('@/views/Dashboard.vue'),
        meta: { perm: 'stats' },
      },
      {
        path: 'models',
        name: 'ModelConfig',
        component: () => import('@/views/ModelConfig.vue'),
        meta: { perm: 'model_config' },
      },
      {
        path: 'voice',
        name: 'VoiceConfig',
        component: () => import('@/views/VoiceConfig.vue'),
        meta: { perm: 'voice_config' },
      },
      {
        path: 'users',
        name: 'UserManage',
        component: () => import('@/views/UserManage.vue'),
        meta: { perm: 'user_manage' },
      },
      {
        path: 'roles',
        name: 'RoleManage',
        component: () => import('@/views/RoleManage.vue'),
        meta: { perm: 'role_manage' },
      },
    ]
  },
  {
    path: '/welcome',
    name: 'Welcome',
    component: () => import('@/views/Welcome.vue'),
    meta: { public: true }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// 所有需要权限的菜单路由定义（按优先级顺序）
const MENU_ROUTES = [
  { name: 'Chat',          path: '/chat',       perm: 'chat' },
  { name: 'KnowledgeBase', path: '/kb',          perm: 'kb_manage' },
  { name: 'Dashboard',     path: '/dashboard',  perm: 'stats' },
  { name: 'ModelConfig',   path: '/models',     perm: 'model_config' },
  { name: 'VoiceConfig',   path: '/voice',      perm: 'voice_config' },
  { name: 'UserManage',    path: '/users',      perm: 'user_manage' },
  { name: 'RoleManage',    path: '/roles',      perm: 'role_manage' },
]

function findFirstAccessibleRoute(auth) {
  for (const r of MENU_ROUTES) {
    if (auth.hasPermission(r.perm)) {
      return { path: r.path }
    }
  }
  return { name: 'Welcome' }
}

router.beforeEach((to) => {
  const auth = useAuthStore()

  if (to.meta.public) return

  if (!auth.token) {
    return { name: 'Login' }
  }

  if (auth.hasNoRole()) {
    return { name: 'Welcome' }
  }

  // 检查菜单权限：若用户缺少该路由权限，跳转到其有权限访问的第一个菜单
  if (to.meta.perm && !auth.hasPermission(to.meta.perm)) {
    return findFirstAccessibleRoute(auth)
  }
})

export default router
