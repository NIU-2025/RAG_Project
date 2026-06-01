/**
 * 上传 Web Worker
 * 文件读取和 XHR 上传全在 Worker 线程执行，不阻塞主线程。
 * 调用方通过 postMessage 传入 { file, kbId, token, uploadUrl }，
 * Worker 上传完成后通过 postMessage 返回 { success, data } 或 { error }。
 */
self.addEventListener('message', async ({ data }) => {
  const { file, kbId, token, uploadUrl } = data

  try {
    // 文件读取在 Worker 线程中进行，不阻塞主线程
    const arrayBuffer = await file.arrayBuffer()
    const blob = new Blob([arrayBuffer], { type: file.type || 'application/octet-stream' })

    const formData = new FormData()
    formData.append('file', blob, file.name)
    formData.append('kb_id', String(kbId))

    const xhr = new XMLHttpRequest()
    xhr.open('POST', uploadUrl)

    if (token) {
      xhr.setRequestHeader('Authorization', `Bearer ${token}`)
    }

    xhr.send(formData)

    xhr.onload = function () {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          self.postMessage({ success: true, data: JSON.parse(xhr.responseText) })
        } catch {
          self.postMessage({ success: true, data: xhr.responseText })
        }
      } else {
        let detail = '上传失败'
        let status = xhr.status
        try { detail = JSON.parse(xhr.responseText).detail } catch {}
        self.postMessage({ success: false, error: detail, status })
      }
    }

    xhr.onerror = function () {
      self.postMessage({ success: false, error: '网络错误，请检查后端服务是否正常运行' })
    }

    xhr.ontimeout = function () {
      self.postMessage({ success: false, error: '上传超时，请检查网络或减小文件体积' })
    }

  } catch (err) {
    self.postMessage({ success: false, error: err.message || '文件读取失败' })
  }
})
