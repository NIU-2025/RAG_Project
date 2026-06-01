import http from './http'

export const authApi = {
  login: (data) => http.post('/auth/login', data),
  register: (data) => http.post('/auth/register', data),
  getMyPermissions: () => http.get('/auth/me/permissions'),
}

export const roleApi = {
  list: (params) => http.get('/roles', { params }),
  get: (id) => http.get(`/roles/${id}`),
  create: (data) => http.post('/roles', data),
  update: (id, data) => http.put(`/roles/${id}`, data),
  delete: (id) => http.delete(`/roles/${id}`),
  getAllPermissions: () => http.get('/roles/permissions/all'),
  updateMenuPermissions: (id, permission_ids) => http.put(`/roles/${id}/menu-permissions`, { permission_ids }),
  updatePermissions: (id, data) => http.put(`/roles/${id}/permissions`, data),
  updateKbPermissions: (id, kb_ids) => http.put(`/roles/${id}/kb-permissions`, { kb_ids }),
}

export const userApi = {
  list: (params) => http.get('/users', { params }),
  get: (id) => http.get(`/users/${id}`),
  delete: (id) => http.delete(`/users/${id}`),
  assignRoles: (id, data) => http.post(`/users/${id}/roles`, data),
  getAllRoles: () => http.get('/users/roles/all'),
}

export const kbApi = {
  list: (params) => http.get('/kb', { params }),
  create: (data) => http.post('/kb', data),
  get: (id) => http.get(`/kb/${id}`),
  update: (id, data) => http.put(`/kb/${id}`, data),
  delete: (id) => http.delete(`/kb/${id}`),
}

export const docsApi = {
  list: (params) => http.get('/docs', { params }),
  upload: (formData) => http.post('/docs/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
  delete: (id) => http.delete(`/docs/${id}`),
  reprocess: (id) => http.post(`/docs/${id}/reprocess`),
}

export const chatApi = {
  createSession: (data) => http.post('/chat/sessions', data),
  listSessions: (params) => http.get('/chat/sessions', { params }),
  getMessages: (sessionId) => http.get(`/chat/sessions/${sessionId}/messages`),
  deleteSession: (sessionId) => http.delete(`/chat/sessions/${sessionId}`),
  sendMessage: (data) => http.post('/chat/completions', data),
}

export const modelsApi = {
  list: () => http.get('/models'),
  test: (provider) => http.post(`/models/${provider}/test`),
  upsert: (provider, data) => http.put(`/models/${provider}`, data),
  remove: (provider) => http.delete(`/models/${provider}`),
}

export const voiceApi = {
  list: () => http.get('/voice'),
  upsert: (provider, data) => http.put(`/voice/${provider}`, data),
  remove: (provider) => http.delete(`/voice/${provider}`),
}

export const statsApi = {
  overview: () => http.get('/stats/overview'),
  kbs: () => http.get('/stats/kbs'),
  parse: () => http.get('/stats/parse'),
  chatDaily: (days) => http.get('/stats/chat/daily', { params: { days } }),
}
