# RAG 检索增强生成系统

基于 FastAPI + ChromaDB + MySQL 的企业级 RAG 知识库问答系统，支持多模态文档解析、混合检索、Cross-Encoder 重排序、多轮对话改写与 RAGAS 评估体系。

## 系统架构

```
用户请求
    │
    ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Query       │───▶│  混合检索         │───▶│  Cross-Encoder   │
│  改写(可选)   │    │  (向量 + BM25)    │    │  重排序           │
└──────────────┘    └──────────────────┘    └──────────────────┘
                                                    │
                                                    ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  LLM 生成    │◀───│  上下文组装       │◀───│  Top-K 精选      │
│  回答        │    │  (Prompt 构建)    │    │                  │
└──────────────┘    └──────────────────┘    └──────────────────┘
```

## 核心功能

### 文档解析
- **PDF**：文本提取 + 表格提取（pdfplumber） + 图片 OCR（RapidOCR）
- **Word / Excel / CSV / TXT / MD**：全格式支持
- **图片**：OCR 文字识别
- **语音**：百度 / 阿里云 ASR 语音转文字

### 检索体系
- **向量检索**：BGE-M3 Embedding → ChromaDB（HNSW 索引）
- **BM25 检索**：jieba 分词 + rank-bm25，Redis 缓存语料
- **动态权重融合**：根据查询特征自动调节向量/BM25 权重
- **Cross-Encoder 重排序**：BGE-Reranker-V2-M3 精排，sigmoid 归一化

### 多轮对话
- **Query 改写**：三层过滤（无历史跳过 → 规则检测 → LLM 改写）
- **流式输出**：SSE 流式响应
- **会话管理**：MySQL 持久化对话历史

### 评估体系
- **RAGAS ContextPrecision**：LLM Judge 语义级检索质量评估
- **网格搜索**：自动搜索最优向量/BM25 权重组合
- **对比评估**：不同检索策略的横向对比

### 其他
- **RBAC 权限**：用户-角色-权限三级权限模型
- **速率限制**：基于 slowapi 的接口限流
- **Redis 缓存**：BM25 语料、权限数据多级缓存
- **GPU 加速**：Reranker 支持 GPU 推理（FP16）

## 快速启动

### 前置条件

- Python 3.11+
- MySQL 8.0+
- Redis 7.0+（可选，用于缓存加速）
- 4GB+ 内存（CPU 推理）或 NVIDIA GPU 4GB+ 显存（GPU 加速）

### 1. 后端

```bash
# 克隆项目
cd RAGProject

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填写 MySQL 密码和 API Key

# MySQL 建库
mysql -u root -p -e "CREATE DATABASE rag_system CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# 启动（自动建表 + 预加载模型）
python main.py
```

API 文档：http://localhost:8000/docs

### 2. 前端

```bash
cd frontend

npm install
npm run dev
```

前端地址：http://localhost:5173

前端页面展示
<img width="1879" height="867" alt="image" src="https://github.com/user-attachments/assets/b9a7c488-2e4f-4179-a745-4ac32a3bd179" />

<img width="1869" height="838" alt="image" src="https://github.com/user-attachments/assets/a859d3c9-0c7f-4010-a355-b769f4d44762" />
<img width="1895" height="873" alt="image" src="https://github.com/user-attachments/assets/942ea2d8-60af-4e4e-b125-28267ae384a4" />




## 目录结构

```
RAGProject/
├── app/
│   ├── api/
│   │   ├── routes/          # FastAPI 路由
│   │   │   ├── auth.py      # 注册/登录/JWT
│   │   │   ├── kb.py        # 知识库 CRUD
│   │   │   ├── docs.py      # 文档上传/管理
│   │   │   ├── chat.py      # 对话/流式生成
│   │   │   ├── search.py    # 检索接口
│   │   │   ├── models.py    # 模型配置
│   │   │   ├── voice.py     # 语音配置
│   │   │   ├── stats.py     # 统计看板
│   │   │   ├── user.py      # 用户管理
│   │   │   └── role.py      # 角色权限
│   │   └── deps.py          # 依赖注入（认证/权限）
│   ├── core/
│   │   ├── config.py        # Pydantic 配置（.env 映射）
│   │   ├── security.py      # JWT + 密码哈希
│   │   ├── logger.py        # Loguru 日志
│   │   └── redis_client.py  # Redis 异步客户端
│   ├── db/
│   │   ├── session.py       # SQLAlchemy 异步引擎
│   │   └── vector_store.py  # ChromaDB 封装
│   ├── models/
│   │   ├── db.py            # SQLAlchemy ORM 模型
│   │   └── schemas.py       # Pydantic 请求/响应模型
│   ├── parsers/
│   │   ├── base.py          # 解析器基类
│   │   ├── pdf.py           # PDF 解析（文本+表格+OCR）
│   │   ├── word.py          # Word 解析
│   │   ├── excel.py         # Excel/CSV 解析
│   │   ├── text.py          # TXT/MD 解析
│   │   ├── image.py         # 图片 OCR
│   │   ├── voice.py         # 语音解析
│   │   └── ocr_utils.py     # OCR 工具函数
│   └── services/
│       ├── embedding.py     # Embedding 推理（BGE-M3）
│       ├── retrieval.py     # 混合检索 + 动态权重
│       ├── reranker.py      # Cross-Encoder 重排序
│       ├── query_rewriter.py# 多轮对话改写
│       ├── llm.py           # LLM 调用（多供应商）
│       ├── document.py      # 文档处理流水线
│       ├── ocr.py           # OCR 服务
│       └── voice_asr.py     # 语音识别服务
├── tests/
│   └── evaluation/
│       ├── evaluate_rag.py      # RAGAS 评估
│       ├── evaluate_weights.py  # 权重网格搜索
│       ├── compare_eval.py      # 对比评估
│       └── eval_dataset.json    # 测试数据集
├── frontend/               # Vue 3 前端
├── uploads/                # 上传文件存储
├── chroma_db/              # ChromaDB 向量持久化
├── logs/                   # 运行日志
├── .env                    # 环境变量（不提交）
├── main.py                 # 应用入口
└── requirements.txt        # Python 依赖
```

## 配置说明

核心配置项（`.env`）：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `EMBEDDING_PROVIDER` | Embedding 方式：`local` / `openai` | `local` |
| `EMBEDDING_MODEL` | 本地 Embedding 模型 | `BAAI/bge-m3` |
| `DEFAULT_LLM_PROVIDER` | LLM 供应商：`deepseek` / `openai` / `dashscope` / `qianfan` / `ollama` | `deepseek` |
| `VECTOR_WEIGHT` | 向量检索基础权重 | `0.7` |
| `BM25_WEIGHT` | BM25 检索基础权重 | `0.3` |
| `RERANK_ENABLED` | 是否启用重排序 | `True` |
| `RERANK_MODEL` | 重排序模型 | `BAAI/bge-reranker-v2-m3` |
| `RETRIEVAL_TOP_K` | 检索返回数量 | `5` |
| `CHUNK_SIZE` | 文档分块大小 | `512` |
| `REDIS_URL` | Redis 连接地址 | `redis://127.0.0.1:6379/0` |

## 检索流程详解

```
用户查询
    │
    ├─▶ Query 改写（多轮对话时触发）
    │     1. 无历史 → 跳过
    │     2. 规则检测 → 零成本过滤
    │     3. LLM 改写 → 仅 ~15-20% 追问触发
    │
    ├─▶ 混合检索
    │     ├─ 向量检索：BGE-M3 → ChromaDB（Top-K × 4）
    │     ├─ BM25 检索：jieba 分词 → rank-bm25（Redis 缓存语料）
    │     └─ 动态权重融合（5 条独立策略叠加）
    │         ① 单侧无结果 → 兜底
    │         ② 精确引用词 → BM25 加分
    │         ③ 具体数量词 → BM25 加分
    │         ④ 语义疑问句 → 向量加分
    │         ⑤ 低重叠度 → 均衡权重
    │
    ├─▶ Cross-Encoder 重排序
    │     └─ BGE-Reranker-V2-M3 → sigmoid 归一化 → Top-K 精选
    │
    └─▶ LLM 生成
          └─ 组装 Prompt → 流式输出回答
```

## RAGAS 评估体系

用于量化评估 RAG 系统检索质量与生成质量，支持基线对比，形成"评估 → 优化 → 对比 → 迭代"闭环。

### 评估指标

全指标基于本地 Embedding 余弦相似度计算（零 LLM 依赖，秒级完成，永不 NaN）：

| 维度 | 指标 | 说明 | 理想值 |
|------|------|------|--------|
| 检索精度 | `context_precision` | 检索到的 chunks 中与 query 语义相关的比例 | > 0.90 |
| 检索召回 | `context_recall` | 是否至少检索到一条相关 chunk | > 0.95 |
| 生成忠实度 | `faithfulness` | 回答中的句子在上下文中有语义支撑的比例 | > 0.85 |
| 回答相关性 | `answer_relevancy` | 回答整体与提问的语义相似度 | > 0.80 |
| 检索召回 | `retrieval/recall@k` | 标注答案在检索结果中被召回的比例 | > 0.80 |
| 检索精度 | `retrieval/precision@k` | 检索结果中与标注答案相关的比例 | > 0.70 |

### 命令速查

| 操作 | 命令 | 输出 |
|------|------|------|
| 运行评估 | `python tests/evaluation/evaluate_rag.py` | `eval_report.md` + `eval_report.json` |
| 对比基线 | `python tests/evaluation/compare_eval.py` | 终端打印 diff |
| 保存新基线 | `python tests/evaluation/evaluate_rag.py --save-baseline` | 覆盖 `baseline.json` |
| 权重搜索 | `python tests/evaluation/evaluate_weights.py` | `weight_eval_report.md` |

### 如何读报告

**Step 1 — 分开看两个维度**

不要混为一谈。先看检索层（context_precision / context_recall / retrieval/*），再看生成层（faithfulness / answer_relevancy）。检索差就调检索参数，生成差就调 prompt 或换 LLM。

**Step 2 — 按类别看弱点**

报告按 6 类查询（事实型、推理型、边界条件、口语化、跨章节综合、无关查询）分别统计，找到最差的类别定点优化。

**Step 3 — 逐条定位**

逐条表标记了每条数据的具体分数，找到最低的几条分析原因。

**Step 4 — 对比基线验证优化**

修改配置或代码后重新运行评估，用 `compare_eval.py` 查看每个指标的变化：

```
context_precision    0.9767 → 0.9800  [+] +0.0033  ← 提升
faithfulness         0.8381 → 0.9100  [+] +0.0719  ← 优化有效
answer_relevancy     0.7830 → 0.7950  [+] +0.0120  ← 轻微提升
```

### 基线管理

- 首次运行 `evaluate_rag.py` 会自动生成 `baseline.json`（如果不存在）
- 后续运行会自动与 `baseline.json` 对比，终端打印每个指标的变化
- 优化后确认效果满意，执行 `--save-baseline` 覆盖旧基线
- 基线文件存储在 `tests/evaluation/baseline.json`，建议随代码一起提交

### 注意事项

- 评估报告的分数是**相对参考值**，不是绝对真理。偶有假阳性（回答正确但指标偏低），需结合人工判断。
- 修改 `CHUNK_SIZE` / `CHUNK_OVERLAP` / `RETRIEVAL_TOP_K` / prompt 后，都需要**重新分块或重启服务**再评估。
- 修改 `.env` 中的 `DEFAULT_LLM_PROVIDER` 后，重启服务再评估。

## 技术栈

| 层级 | 技术 |
|------|------|
| 框架 | FastAPI + Uvicorn |
| 前端 | Vue 3 + Element Plus + Vite |
| 向量库 | ChromaDB（HNSW 索引） |
| 关系库 | MySQL 8.0 + SQLAlchemy 2.0（异步） |
| 缓存 | Redis 7.0（BM25 语料 + 权限缓存） |
| Embedding | BGE-M3（sentence-transformers） |
| Reranker | BGE-Reranker-V2-M3（FlagEmbedding） |
| LLM | DeepSeek / OpenAI / 通义千问 / 文心一言 / Ollama |
| OCR | RapidOCR + ONNX Runtime |
| ASR | 百度 / 阿里云语音识别 |
| 评估 | RAGAS + DeepSeek LLM Judge |
| 认证 | JWT + bcrypt + RBAC |
| 限流 | slowapi |
| 日志 | Loguru |
