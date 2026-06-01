# RAG 检索增强生成系统

## 快速启动

### 1. 后端

```bash
cd rag-system

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填写 MySQL 密码和 API Key

# 启动（开发模式，自动建表）
python main.py
# 或
uvicorn main:app --reload --port 8000
```

API 文档：http://localhost:8000/docs

### 2. 前端

```bash
cd rag-system/frontend

npm install
npm run dev
```

前端地址：http://localhost:5173

### 3. MySQL 建库

```sql
CREATE DATABASE rag_system CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

## 目录说明

```
rag-system/
├── app/
│   ├── api/routes/     # FastAPI 路由（auth/kb/docs/chat/search）
│   ├── core/           # 配置、认证、日志
│   ├── db/             # MySQL Session、ChromaDB 封装
│   ├── models/         # SQLAlchemy 模型 + Pydantic Schema
│   ├── parsers/        # 文档解析（PDF/Word/Excel/CSV/TXT/MD）
│   └── services/       # 业务逻辑（document/embedding/retrieval/llm）
├── frontend/           # Vue 3 前端
├── uploads/            # 上传文件存储
├── chroma_db/          # ChromaDB 向量持久化
├── .env                # 环境变量
└── main.py             # 入口
```
