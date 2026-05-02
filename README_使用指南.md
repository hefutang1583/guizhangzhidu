# 校园规章制度 RAG 问答系统 - 使用指南

## 系统架构
```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   index.html  │────▶│ FastAPI 后端服务  │────▶│ 向量知识库       │
│   (前端页面)   │◀────│  (main.py)       │◀────│ (vector_db/)    │
└──────────────┘     └──────────────────┘     └─────────────────┘
                            │
                     ┌──────┴──────┐
                     │ rag_chain.py │  ← RAG核心（检索+生成）
                     │ + DeepSeek   │  ← 大模型（智能回答）
                     └─────────────┘
```

## 文件说明
| 文件 | 作用 |
|------|------|
| `build_knowledge_base.py` | 知识库构建脚本（加载PDF/DOCX → 清洗 → 分块 → TF-IDF向量化 → 存储） |
| `rag_chain.py` | RAG 核心流程（向量检索 + LLM 生成回答） |
| `main.py` | FastAPI 后端服务器（自动加载 .env 配置） |
| `index.html` | 前端聊天界面 |
| `.env` | 大模型 API Key 配置（需自行创建） |
| `.env.example` | 配置文件模板 |
| `vector_db/campus_rules/` | 构建好的向量知识库数据 |
| `河北大学规章制度/` | 原始文档（8个PDF + 2个DOCX） |

---

## 运行步骤

### 前提条件
- 已安装 **Python 3.9** 或 **Python 3.14**
- 项目路径: `C:\Users\Administrator\Desktop\MyRAGProject`
- 虚拟环境路径: `C:\Users\Administrator\Desktop\MyRAGProject\venv`

### 第一步：打开 CMD（命令提示符）

按 `Win + R`，输入 `cmd`，回车。

### 第二步：进入项目目录

```cmd
cd /d C:\Users\Administrator\Desktop\MyRAGProject
```

### 第三步：激活虚拟环境

虚拟环境已经创建好了，直接执行：

```cmd
venv\Scripts\activate
```

**成功标志**: 命令行前面出现 `(venv)` 提示符，例如：
```
(venv) C:\Users\Administrator\Desktop\MyRAGProject>
```

> 如果提示"无法加载文件"或"执行策略"错误，在CMD中先执行：
> ```cmd
> powershell -Command "Set-ExecutionPolicy -Scope CurrentUser RemoteSigned"
> ```

### 第四步：（首次运行）安装依赖包

如果之前没有安装过依赖：

```cmd
pip install -r requirements.txt
```

> 注意：当前系统使用的是 **Python 3.14 + 自建 venv 虚拟环境**，
> 依赖已全部安装好。如果遇到兼容性问题，建议用 Python 3.9 重新创建虚拟环境：
> 
> ```cmd
> :: 使用 Python 3.9 创建新的虚拟环境
> py -3.9 -m venv venv39
> venv39\Scripts\activate
> pip install -r requirements.txt
> ```

### 第五步：配置大语言模型（★ 重要）

系统需要大语言模型来生成有逻辑的回答。**推荐使用 DeepSeek**（国产，便宜，效果好）。

#### 5.1 获取 DeepSeek API Key

1. 打开 https://platform.deepseek.com
2. 注册账号（手机号即可）
3. 登录后进入 **API Keys** 页面
4. 点击 **创建 API Key**，复制生成的 Key（以 `sk-` 开头）
5. 充值（最低 10 元，可使用很久。输入约 ¥0.001/千字，输出约 ¥0.002/千字）

#### 5.2 创建配置文件

在项目根目录创建 `.env` 文件：

```cmd
:: 在 CMD 中执行（确保在项目目录下）
echo DEEPSEEK_API_KEY=sk-你的API密钥> .env
echo DEEPSEEK_MODEL=deepseek-chat>> .env
echo DEEPSEEK_BASE_URL=https://api.deepseek.com>> .env
```

或者手动创建 `.env` 文件，内容如下：
```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

> **如果不配置 API Key**：系统仍可运行，但会使用"仅检索模式"，
> 回答只是相关片段的简单罗列，没有逻辑整合。

### 第六步：构建知识库（只需运行一次）

```cmd
python build_knowledge_base.py
```

看到以下输出表示构建成功：
```
==================================================
知识库构建完成!
   documents_loaded: 9
   chunks_created: 183
   vector_dim: 4943
==================================================
```

### 第七步：启动后端服务

保持 CMD 窗口，执行：

```cmd
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

看到以下输出表示启动成功：
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

> 如果配置了 DeepSeek API Key，启动时会在日志中看到：
> `✅ DeepSeek LLM 已接入`

### 第八步：打开前端页面

**方式 A — 直接双击打开（推荐）**:
- 在资源管理器中双击 `C:\Users\Administrator\Desktop\MyRAGProject\index.html`

**方式 B — 浏览器访问**:
- 打开浏览器访问 `http://127.0.0.1:8000/docs` 查看 API 文档

---

## API 接口说明

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查，返回向量库状态 |
| `/api/query` | POST | RAG问答，传入 `{"question": "问题内容"}` |
| `/api/documents` | GET | 列出知识库中所有文档来源 |
| `/docs` | GET | Swagger 自动生成的API文档 |

---

## 常见问题

### Q1: 启动时提示 "模块未找到"
**解决**: 确保已激活虚拟环境（第三步），命令行前应有 `(venv)`。

### Q2: 前端显示 "请求失败: Failed to fetch"
**原因**: 后端服务未启动。
**解决**: 先执行第七步启动后端，再刷新前端页面。
**注意**: 当前版本已修复此问题——即使以 file:// 协议打开 HTML，
也会自动连接到 `http://127.0.0.1:8000`。

### Q3: 如何停止服务？
在运行服务的 CMD 窗口中按 `Ctrl + C` 即可停止。

### Q4: 更新文档后需要重新构建吗？
**是的**。如果修改了 `河北大学规章制度/` 目录下的文件，
需要重新执行第六步：
```cmd
python build_knowledge_base.py
```
然后重启服务。

### Q5: 回答只是简单罗列，没有逻辑？
**原因**: 未配置大语言模型 API Key。
**解决**: 按第五步配置 DeepSeek API Key，然后重启服务。

### Q6: DeepSeek API 调用失败？
- 检查 `.env` 文件是否在项目根目录
- 检查 API Key 是否正确（以 `sk-` 开头）
- 检查网络是否能访问 `https://api.deepseek.com`
- 检查账户余额是否充足

### Q7: 能否使用其他大模型？
可以！支持任何 OpenAI 兼容接口。在 `.env` 中配置：
```
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=你的接口地址
LLM_MODEL=模型名称
```
支持的模型：DeepSeek、OpenAI GPT、通义千问、智谱 GLM 等。

---

## 关于虚拟环境

本项目使用的虚拟环境位于：
```
C:\Users\Administrator\Desktop\MyRAGProject\venv\
```

这是用 Python 3.14 创建的虚拟环境。如果你更倾向于使用 Python 3.9：

```cmd
:: 创建基于 Python 3.9 的虚拟环境
cd C:\Users\Administrator\Desktop\MyRAGProject
py -3.9 -m venv venv39

:: 激活它
venv39\Scripts\activate

:: 安装依赖
pip install -r requirements.txt

:: 之后步骤相同...
```

退出虚拟环境的命令：
```cmd
deactivate
```
