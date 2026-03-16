# NexusRAG Windows 本地配置指南

本文档说明如何在 Windows 系统上（不使用 Docker）配置和运行 NexusRAG 项目。

---

## 目录

- [系统要求](#系统要求)
- [1. 安装前提条件](#1-安装前提条件)
- [2. 项目初始化](#2-项目初始化)
- [3. 环境配置](#3-环境配置)
- [4. 下载 ML 模型](#4-下载-ml-模型)
- [5. 启动服务](#5-启动服务)
- [Windows 常见问题](#windows-常见问题)
- [验证安装](#验证安装)
- [故障排除](#故障排除)

---

## 系统要求

| 资源 | 最低配置 | 推荐配置 |
|------|---------|---------|
| RAM | 4 GB | 8 GB+ |
| 磁盘空间 | 5 GB | 10 GB+ (含模型约 2.5GB) |
| Python | 3.10+ | 3.11+ |
| Node.js | 18+ | 22 LTS |
| 操作系统 | Windows 10/11 | Windows 11 |

---

## 1. 安装前提条件

### 1.1 Python 3.10+

1. 访问 https://www.python.org/downloads/
2. 下载 Python 3.11 或 3.12 的 Windows 安装包
3. **重要**：安装时勾选 **"Add Python to PATH"**（添加 Python 到系统环境变量）
4. 完成安装后，打开新的 PowerShell 或 CMD 验证：
   ```bash
   python --version
   # 应该显示: Python 3.11.x 或更高
   ```

### 1.2 Node.js 18+

1. 访问 https://nodejs.org/
2. 下载 **LTS 版本**（推荐 22.x）
3. 安装时勾选 **"Add to PATH"**
4. 验证安装：
   ```bash
   node --version
   npm --version
   ```

### 1.3 pnpm

pnpm 是前端包管理器，比 npm 更高效：

```bash
npm install -g pnpm
```

验证：
```bash
pnpm --version
```

### 1.4 PostgreSQL（数据库）

**方式 A：直接安装 PostgreSQL**

1. 访问 https://www.postgresql.org/download/windows/
2. 推荐使用 **EnterpriseDB** 安装包
3. 安装过程中：
   - 设置密码：`postgres`（或自定义，记得修改配置）
   - 端口：默认 `5432`
   - 选择安装 pgAdmin（可选，用于图形化管理）
4. 安装完成后，创建数据库：
   - 打开 **pgAdmin** 或使用命令行：
   ```bash
   # 使用 psql 命令行
   psql -U postgres
   CREATE DATABASE nexusrag;
   \q
   ```
5. 配置端口：项目使用端口 **5433**，你需要：
   - 要么修改 PostgreSQL 监听端口为 5433
   - 要么修改项目配置文件中的端口（见第 3 步）

**方式 B：使用 Docker（仅 PostgreSQL）**

如果不想在 Windows 安装 PostgreSQL，可以用 Docker：
```bash
docker run -d \
  --name nexusrag-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=nexusrag \
  -p 5433:5432 \
  postgres:15-alpine
```

### 1.5 ChromaDB（向量数据库）

**方式 A：Python 本地服务器（推荐）**

1. 激活项目虚拟环境后（见第 2 步）：
   ```bash
   pip install chromadb
   ```
2. 启动 ChromaDB 服务器：
   ```bash
   chroma run --host localhost --port 8002 --path chroma_data
   ```
   保持这个终端窗口运行

**方式 B：Docker**
```bash
docker run -d \
  --name nexusrag-chromadb \
  -p 8002:8000 \
  -v chroma_data:/chroma/chroma \
  -e IS_PERSISTENT=TRUE \
  -e ANONYMIZED_TELEMETRY=FALSE \
  chromadb/chroma:latest
```

### 1.6 Ollama（可选 - 本地 LLM）

如果不想使用 Gemini API，可以使用本地 LLM：

1. 访问 https://ollama.ai/download/windows
2. 下载并安装 Windows 版本
3. 安装完成后，Ollama 会在后台运行（系统托盘显示图标）
4. 下载模型（首次使用）：
   ```bash
   # 在任意终端运行（不需要激活虚拟环境）
   ollama pull gemma3:12b
   # 或使用更小的模型
   ollama pull qwen3:4b
   ```

---

## 2. 项目初始化

### 2.1 克隆项目

```bash
# 如果还没有克隆
git clone https://github.com/LeDat98/NexusRAG.git
cd NexusRAG
```

### 2.2 创建 Python 虚拟环境

```bash
# PowerShell 或 CMD
python -m venv venv
```

### 2.3 激活虚拟环境

**PowerShell（推荐）**：
```powershell
# 如果遇到执行策略错误，先运行：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 然后激活
venv\Scripts\Activate.ps1
```

**CMD**：
```cmd
venv\Scripts\activate.bat
```

激活后，命令行前面应该显示 `(venv)`。

### 2.4 安装 Python 依赖

```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

⚠️ **注意**：首次安装会下载约 2.5GB 的机器学习模型（sentence-transformers），可能需要几分钟到几十分钟，取决于网络速度。如果失败，重试或使用镜像源。

---

## 3. 环境配置

### 3.1 复制环境变量文件

```bash
copy .env.example .env
```

### 3.2 编辑 `.env` 文件

用任意文本编辑器（推荐 VS Code 或记事本）打开 `.env` 文件。

**必填配置项：**

#### 选项 1：使用 Gemini（云端 API）

```
# 语言模型提供商
LLM_PROVIDER=gemini

# Google AI API 密钥（必须）
# 获取地址：https://aistudio.google.com/app/apikey
GOOGLE_AI_API_KEY=你的API密钥
```

#### 选项 2：使用 Ollama（本地运行）

```
# 语言模型提供商
LLM_PROVIDER=ollama

# Ollama 服务地址（默认）
OLLAMA_HOST=http://localhost:11434

# 使用的模型名（需要提前 ollama pull）
OLLAMA_MODEL=gemma3:12b
```

**知识图谱嵌入配置（可选，建议保持默认）：**

```
# KG 嵌入提供商（可与 LLM 不同）
KG_EMBEDDING_PROVIDER=sentence_transformers
KG_EMBEDDING_MODEL=BAAI/bge-m3
KG_EMBEDDING_DIMENSION=1024
```

使用 `sentence_transformers` 可以完全本地运行，无需 API。

**数据库配置（如果端口不同）：**

如果 PostgreSQL 运行在默认端口 5432 而不是 5433：
```bash
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/nexusrag
```

如果 ChromaDB 运行在其他端口：
```bash
CHROMA_HOST=localhost
CHROMA_PORT=8002  # 改为实际端口
```

**CORS 配置**（默认即可）：
```
CORS_ORIGINS=["http://localhost:5174","http://localhost:3000"]
```

---

## 4. 下载 ML 模型

虽然模型会在首次使用时自动下载，但建议提前下载以避免运行时延迟：

```bash
python backend/scripts/download_models.py
```

这将下载：
- `BAAI/bge-m3` - 嵌入模型（~1.4 GB）
- `BAAI/bge-reranker-v2-m3` - 重排序模型（~1.1 GB）

模型文件会存储在 `~/.cache/torch/sentence_transformers/` 目录。

---

## 5. 启动服务

需要 **3-4 个终端窗口** 同时运行：

---

### 终端 1：PostgreSQL 服务

**如果作为 Windows 服务安装**（默认）：

1. 按 `Win + R`，输入 `services.msc`
2. 找到 `postgresql-x64-15`（版本号可能不同）
3. 右键启动，并设置启动类型为"自动"

**如果使用 Docker**：
```bash
docker start nexusrag-postgres  # 或 docker run（首次）
```

**如果手动启动**：
```bash
# 进入 PostgreSQL 安装目录，例如：
"C:\Program Files\PostgreSQL\15\bin\pg_ctl.exe" start -D "C:\Program Files\PostgreSQL\15\data"
```

---

### 终端 2：ChromaDB 服务

```bash
# 激活虚拟环境
venv\Scripts\activate

# 启动 ChromaDB 服务器
chroma run --host localhost --port 8002 --path chroma_data
```

✅ 应该看到输出：`ChromaDB running in HTTP mode`，保持运行。

---

### 终端 3：后端服务（FastAPI）

```bash
cd D:\llm_rag\NexusRAG
venv\Scripts\activate
cd backend
uvicorn app.main:app --reload --port 8080
```

✅ 成功启动后看到：
```
Uvicorn running on http://127.0.0.1:8080
```

API 文档会自动生成：http://localhost:8080/docs

---

### 终端 4：前端服务（React + Vite）

```bash
cd D:\llm_rag\NexusRAG
pnpm install  # 首次运行
pnpm dev
```

✅ 成功启动后输出：
```
VITE v.x.x.x  ready in xxx ms

➜  Local:   http://localhost:5174/
```

---

## Windows 常见问题

### Q1: `python` 命令找不到

确保在安装 Python 时勾选了 "Add Python to PATH"，然后重启终端。或者使用完整路径：
```bash
C:\Users\你的用户名\AppData\Local\Programs\Python\Python311\python.exe -m venv venv
```

### Q2: PowerShell 激活虚拟环境报错

```powershell
# 错误：无法加载文件，因为在此系统上禁止运行脚本
# 解决：修改执行策略
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

然后重试激活。

### Q3: `chroma` 命令找不到

确保在激活的虚拟环境中安装了 chromadb：
```bash
venv\Scripts\activate
pip install chromadb
```

如果还是找不到，使用完整路径：
```bash
venv\Scripts\python.exe -m chromadb
```

### Q4: PostgreSQL 连接被拒绝 (Connection refused)

检查：
1. PostgreSQL 服务是否在运行：
   ```powershell
   Get-Service -Name postgresql*
   ```
2. 确认端口：`.env` 中的 `DATABASE_URL` 是否与 PostgreSQL 实际端口一致
3. 确认数据库 `nexusrag` 已创建

### Q5: ChromaDB 端口被占用

8002 端口可能被其他程序占用。解决方案：

1. 查找占用端口的进程：
   ```powershell
   netstat -ano | findstr :8002
   ```
2. 结束该进程，或修改 `.env` 中的 `CHROMA_PORT` 为其他端口（如 8003），并相应启动：
   ```bash
   chroma run --port 8003
   ```

### Q6: 前端 `pnpm install` 很慢或失败

1. 配置 pnpm mirror（国内用户）：
   ```bash
   pnpm config set registry https://registry.npmmirror.com/
   ```
2. 如果仍然失败，尝试使用 npm：
   ```bash
   npm install
   npm run dev
   ```

### Q7: 后端启动时报 ImportError

通常是某些依赖安装失败。确保：
1. 虚拟环境已激活
2. requirements.txt 中的包全部安装成功
3. Python 版本 >= 3.10

重新安装：
```bash
pip install --force-reinstall -r backend/requirements.txt
```

### Q8: Model download 慢或失败

sentence-transformers 首次导入会自动下载模型（约 2.5GB）。如果网络慢：

1. **使用科学上网**
2. **设置镜像**（手动下载后放到缓存目录）：
   - 模型缓存目录：`C:\Users\你的用户名\.cache\torch\sentence_transformers`
   - 从国内镜像站下载 `.bin` 文件放入对应文件夹

---

## 验证安装

### 1. 检查后端 API

访问 http://localhost:8080/docs

应该看到 **Swagger UI** 页面，显示 NexusRAG 的所有 API 端点。

点击 `GET /api/v1/workspaces` → **Try it out** → Execute，如果返回 `[]` 说明 API 正常。

### 2. 检查前端页面

访问 http://localhost:5174

应该看到 NexusRAG 的界面（暗色/亮色主题选择）。

### 3. 完整功能测试

1. 在右上角选择或创建工作区
2. 上传一个 PDF 文档（测试文档）
3. 等待文档处理完成（状态从"处理中"变为"完成"）
4. 在聊天框输入问题，如"这份文档讲了什么？"
5. 观察是否有回答、引用（`[a3z1]` 格式）、知识图谱等

---

## 故障排除

### 问题：后端启动时报数据库迁移错误

运行数据库迁移：

```bash
cd backend
alembic upgrade head
```

如果 alembic 未安装或配置错误：
```bash
pip install alembic
alembic upgrade head
```

### 问题：ChromaDB 启动报错"persistence directory not writable"

确保有权限写入 `chroma_data` 目录。删除该目录后重试：
```bash
rmdir /s chroma_data
chroma run --host localhost --port 8002 --path chroma_data
```

### 问题：上传文档时报"docling"相关错误

Docling 依赖一些系统库。确保：
1. 安装了最新版本：`pip install --upgrade docling`
2. 如果处理 PDF 失败，可能需要安装 `poppler`：
   - 下载：https://github.com/oschwin106/poppler-windows/releases/
   - 解压后将 `bin` 目录添加到系统 PATH

### 问题：搜索返回空结果或低相关性

1. 确保文档已完全处理（查看文档列表的状态）
2. 尝试降低相关性阈值：在 `.env` 中添加：
   ```
   NEXUSRAG_MIN_RELEVANCE_SCORE=0.05
   ```
3. 检查 ChromaDB 是否有数据：http://localhost:8002

### 问题：Gemini API 报错

1. 确认 API Key 有效：访问 https://aistudio.google.com/app/apikey
2. 检查 API 配额是否耗尽
3. 确保 `GOOGLE_AI_API_KEY` 在 `.env` 设置正确且无空格

### 问题：Windows 防火墙阻止连接

如果无法访问 `localhost:8080` 或 `localhost:5174`：

1. 允许 Python 和 Node.js 通过防火墙
2. 或暂时关闭防火墙测试：
   ```powershell
   Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False
   ```

---

## 生产环境注意事项（可选）

本地开发使用的是轻量级 ChromaDB 服务器和 PostgreSQL，适合单用户或小团队。如果多用户或生产部署：

1. **PostgreSQL**：使用 AWS RDS 或自建集群，调整 `pool_size`
2. **ChromaDB**：使用生产托管版或自建高可用集群
3. **后端**：使用 Gunicorn + Uvicorn workers 而非 `--reload`
   ```bash
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b :8080
   ```
4. **前端**：使用 `pnpm build` 构建静态文件，用 Nginx 托管
5. **HTTPS**：配置反向代理（Nginx/Traefik）启用 HTTPS
6. **日志**：配置结构化日志收集（JSON 格式）

---

## 相关文件

- `README.md` - 项目整体介绍和功能说明
- `.env.example` - 所有可用配置项的说明
- `backend/requirements.txt` - Python 依赖列表
- `frontend/package.json` - 前端依赖和脚本
- `docker-compose.yml` - Docker 部署配置（供参考）

---

## 获取帮助

- 项目主页：https://github.com/LeDat98/NexusRAG
- 问题反馈：https://github.com/LeDat98/NexusRAG/issues
- 技术文档：查看 `README.md` 中的功能详述部分

---

**祝你使用愉快！** 🚀
