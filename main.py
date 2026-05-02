"""
FastAPI 后端服务 - 校园规章制度咨询助手
提供健康检查、问答接口、文档列表等 API
"""

import os
import sys
import logging
import traceback
from contextlib import asynccontextmanager

# 确保项目目录在 path 中（uvicorn 启动时需要）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 自动加载 .env 配置文件（API Key 等敏感信息）
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str


class HealthResponse(BaseModel):
    status: str
    vector_count: int = 0
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 - 启动时检查知识库"""
    try:
        from rag_chain import load_vector_db
        db, _ = load_vector_db()
        app.state.vector_count = db.count
        app.state.kb_ready = True
        logger.info(f"知识库已加载，共 {db.count} 条记录")
    except Exception as e:
        logger.warning(f"知识库未就绪（可先运行 build_knowledge_base.py 构建）: {e}")
        app.state.vector_count = 0
        app.state.kb_ready = False
    yield


app = FastAPI(
    title="校园规章制度咨询助手 API",
    description="基于 RAG 技术的智能问答系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口，返回系统状态和向量库记录数"""
    status = "ok" if getattr(app.state, "kb_ready", False) else "warning"
    message = "系统运行正常" if getattr(app.state, "kb_ready", False) else "知识库未初始化"
    vc = getattr(app.state, "vector_count", 0)
    return HealthResponse(status=status, vector_count=int(vc), message=message)


@app.post("/api/query")
async def query_knowledge_base(req: QueryRequest):
    """
    RAG 问答核心接口
    
    接收用户问题 -> 检索向量库 -> 返回回答和参考来源
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    question = req.question.strip()
    
    try:
        from rag_chain import query_rag
        
        # ── 大模型初始化 ──
        # 优先级：DEEPSEEK_API_KEY > OPENAI_API_KEY
        # DeepSeek：注册 https://platform.deepseek.com 获取 API Key
        llm = None
        deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
        openai_key = os.getenv("OPENAI_API_KEY", "")

        try:
            from langchain_openai import ChatOpenAI

            if deepseek_key:
                llm = ChatOpenAI(
                    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                    api_key=deepseek_key,
                    temperature=0.3,
                    max_tokens=1024,
                )
                logger.info("✅ DeepSeek LLM 已接入")
            elif openai_key:
                llm = ChatOpenAI(
                    model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
                    base_url=os.getenv("OPENAI_BASE_URL"),
                    api_key=openai_key,
                    temperature=0.3,
                    max_tokens=1024,
                )
                logger.info("✅ OpenAI LLM 已接入")
            else:
                logger.warning("⚠️ 未配置 LLM API Key，使用仅检索模式")
        except Exception as e:
            logger.warning(f"LLM 初始化失败，使用检索模式: {e}")

        result = query_rag(question, llm=llm, top_k=5)

        # 返回简短的参考来源（仅文件名+类型）
        display_sources = []
        for s in result["sources"]:
            display_sources.append({
                "document": s["document"],
                "type": s.get("type", "kb"),
            })

        return {
            "success": True,
            "answer": result["answer"],
            "sources": display_sources,
            "question": question,
        }

    except ValueError as ve:
        logger.error(f"知识库未就绪: {ve}")
        raise HTTPException(status_code=503, detail=f"知识库未就绪: {str(ve)}")
    except Exception as e:
        logger.error(f"查询失败: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@app.get("/api/documents")
async def list_documents():
    """列出知识库中的所有文档来源"""
    try:
        from rag_chain import load_vector_db
        db, _ = load_vector_db()
        
        sources = set()
        for meta in db.metadatas:
            src = meta.get("source", "未知")
            sources.add(src)

        return {"success": True, "documents": sorted(sources), "total_chunks": db.count}
    except Exception as e:
        return {"success": False, "documents": [], "error": str(e)}


@app.get("/", response_class=HTMLResponse)
async def root():
    """返回前端聊天页面"""
    html_path = os.path.join(BASE_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
