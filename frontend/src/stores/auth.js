import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { authApi } from '@/api'

// 所有需要权限的菜单路由定义（按优先级顺序）
const MENU_ROUTES = [
  { path: '/chat',      perm: 'chat' },
  { path: '/kb',        perm: 'kb_manage' },
  { path: '/dashboard', perm: 'stats' },
  { path: '/models',    perm: 'model_config' },
  { path: '/voice',     perm: 'voice_config' },
  { path: '/users',     perm: 'user_manage' },
  { path: '/roles',     perm: 'role_manage' },
]

export function findFirstAccessibleRoute(menuPermissionCodes, isAdmin) {
  for (const r of MENU_ROUTES) {
    if (isAdmin || menuPermissionCodes.includes(r.perm)) {
      return r.path
    }
  }
  return '/welcome'
}

export const useAuthStore = defineStore('auth', () => {
  const token = ref(localStorage.getItem('token') || '')
  const user = ref(JSON.parse(localStorage.getItem('user') || 'null'))
  const roles = ref(JSON.parse(localStorage.getItem('roles') || '[]'))
  const menuPermissionCodes = ref(JSON.parse(localStorage.getItem('menu_perms') || '[]'))
  const kbPermissionIds = ref(JSON.parse(localStorage.getItem('kb_perms') || '[]'))
  const allKbs = ref(JSON.parse(localStorage.getItem('all_kbs') || '[]'))
  const isAdmin = ref(localStorage.getItem('is_admin') === 'true')

  const isLoggedIn = computed(() => !!token.value)

  async function login(username, password) {
    const res = await authApi.login({ username, password })
    token.value = res.access_token
    user.value = res.user
    localStorage.setItem('token', res.access_token)
    localStorage.setItem('user', JSON.stringify(res.user))
    await fetchPermissions()
  }

  async function fetchPermissions() {
    try {
      const raw = await authApi.getMyPermissions()
      // API 返回 { code, message, data: { roles, is_admin, ... } }，需要取 data 层
      const res = raw?.data || raw
      roles.value = res.roles || []
      menuPermissionCodes.value = res.menu_permission_codes || []
      kbPermissionIds.value = res.kb_permission_ids || []
      allKbs.value = res.all_kbs || []
      isAdmin.value = res.is_admin || false
      localStorage.setItem('is_admin', String(isAdmin.value))
      localStorage.setItem('roles', JSON.stringify(roles.value))
      localStorage.setItem('menu_perms', JSON.stringify(menuPermissionCodes.value))
      localStorage.setItem('kb_perms', JSON.stringify(kbPermissionIds.value))
      localStorage.setItem('all_kbs', JSON.stringify(allKbs.value))
    } catch (e) {
      // 忽略
    }
  }

  function hasPermission(permCode) {
    if (isAdmin.value) return true
    return menuPermissionCodes.value.includes(permCode)
  }

  function hasKbAccess(kbId) {
    if (isAdmin.value) return true
    return kbPermissionIds.value.includes(kbId)
  }

  function hasNoRole() {
    return roles.value.length === 0 && !isAdmin.value
  }

  function logout() {
    token.value = ''
    user.value = null
    roles.value = []
    menuPermissionCodes.value = []
    kbPermissionIds.value = []
    allKbs.value = []
    isAdmin.value = false
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    localStorage.removeItem('roles')
    localStorage.removeItem('menu_perms')
    localStorage.removeItem('kb_perms')
    localStorage.removeItem('all_kbs')
    localStorage.removeItem('is_admin')
  }

  return {
    token,
    user,
    roles,
    menuPermissionCodes,
    kbPermissionIds,
    allKbs,
    isAdmin,
    isLoggedIn,
    hasPermission,
    hasKbAccess,
    hasNoRole,
    login,
    logout,
    fetchPermissions,
  }
})
