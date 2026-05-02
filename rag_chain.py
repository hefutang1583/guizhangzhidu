"""
RAG 核心流程模块
- 向量检索（TF-IDF + 余弦相似度）
- LLM 生成回答
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ─── 配置 ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
COLLECTION_NAME = "campus_rules"


# ══════════════════════════════════════════════════
# 向量化器加载（从JSON重建，避免pickle问题）
# ══════════════════════════════════════════════════

def _load_vectorizer():
    """
    从 vectorizer.json 加载并重建 TfidfVectorizer 对象。
    完全不使用 pickle，解决 uvicorn 模块路径问题。
    """
    # 确保项目目录在 path 中
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    import build_knowledge_base as _bk

    vec_path = os.path.join(VECTOR_DB_DIR, COLLECTION_NAME, "vectorizer.json")
    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"找不到向量化模型文件: {vec_path}")

    with open(vec_path, 'r', encoding='utf-8') as f:
        vec_data = json.load(f)

    # 重建 TfidfVectorizer 对象
    vec = _bk.TfidfVectorizer()
    vec.vocabulary = vec_data["vocabulary"]
    vec.idf = {k: float(v) for k, v in vec_data["idf"].items()}
    vec._fitted = True

    logger.info(f"向量器加载成功，词汇量: {len(vec.vocabulary)}")
    return vec


def load_vector_db():
    """加载向量库和向量化器，返回 (db, vectorizer) 元组"""
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    import build_knowledge_base as _bk

    db = _bk.SimpleVectorDB(VECTOR_DB_DIR, COLLECTION_NAME)
    if not db._load():
        raise ValueError("知识库未初始化。请先运行: python build_knowledge_base.py")

    vectorizer = _load_vectorizer()
    return db, vectorizer


# ══════════════════════════════════════════════════
# 向量检索
# ══════════════════════════════════════════════════

def query_vector_db(question: str, collection=None,
                    vectorizer=None, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    查询向量库，返回相关文档片段
    策略：TF-IDF 向量检索 + 关键词匹配互补，再按文档聚合扩展
    """
    if collection is None or vectorizer is None:
        raise ValueError("向量库或向量化器未初始化")
    
    import numpy as np
    import re

    # ── 策略1: TF-IDF 向量检索 ──
    query_vec = vectorizer.transform([question])[0]
    initial_results = collection.query(
        query_embedding=query_vec,
        top_k=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hit_sources = set()
    for meta in initial_results["metadatas"][0]:
        hit_sources.add(meta.get("source", ""))

    # ── 策略2: 关键词匹配（补充 TF-IDF 的不足）──
    # 提取2-4字的中文关键词（滑动窗口）
    cn_chars = re.findall(r'[\u4e00-\u9fff]', question)
    keywords = set()
    for length in [4, 3, 2]:
        for i in range(len(cn_chars) - length + 1):
            keywords.add(''.join(cn_chars[i:i+length]))
    all_docs = collection.documents
    all_metas = collection.metadatas

    for doc_text, meta in zip(all_docs, all_metas):
        src = meta.get("source", "")
        # 至少1个3字以上关键词命中
        match_count = sum(1 for kw in keywords if len(kw) >= 3 and kw in doc_text)
        if match_count >= 1:
            hit_sources.add(src)

    # ── 按文档聚合：合并同一文档的所有片段 ──
    sources = []
    seen = set()
    for doc_text, meta in zip(all_docs, all_metas):
        src = meta.get("source", "")
        if src in hit_sources and src not in seen:
            seen.add(src)
            combined = []
            for d, m in zip(all_docs, all_metas):
                if m.get("source", "") == src:
                    combined.append(d)
            full_text = "\n".join(combined)
            # 只保留文件名（去掉路径和哈希前缀）
            import os
            display_name = os.path.basename(src)
            # 去掉常见的哈希前缀模式（如 FF8725CD61D6E2F5CD2919C0821_）
            import re
            display_name = re.sub(r'^[A-F0-9]{20,}_', '', display_name)
            sources.append({
                "document": display_name,
                "snippet": full_text[:4000],  # 内部使用，不返回给前端
                "score": 1.0,
                "type": "kb",
            })

    logger.info(f"检索到 {len(sources)} 个相关文档 (命中来源: {hit_sources})")
    return sources


# ══════════════════════════════════════════════════
# 联网搜索（DuckDuckGo，免费无需API Key）
# ══════════════════════════════════════════════════

def web_search(question: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """使用 DuckDuckGo 搜索引擎获取网络结果（多后端回退）"""
    # 优先使用新版 ddgs 包
    ddgs_classes = []
    try:
        from ddgs import DDGS as DDGS_new
        ddgs_classes.append(("ddgs", DDGS_new))
    except ImportError:
        pass
    try:
        from duckduckgo_search import DDGS as DDGS_old
        ddgs_classes.append(("duckduckgo_search", DDGS_old))
    except ImportError:
        pass

    if not ddgs_classes:
        logger.warning("未安装搜索库，请运行: pip install ddgs")
        return []

    for pkg_name, DDGS in ddgs_classes:
        for backend in ["duckduckgo", "auto"]:
            try:
                with DDGS() as ddgs:
                    if pkg_name == "ddgs":
                        # 新版 ddgs: text(query, region=, max_results=, backend=)
                        results = list(ddgs.text(question, region="cn-zh",
                                                  max_results=max_results, backend=backend))
                    else:
                        # 旧版 duckduckgo_search: text(keywords=, region=, max_results=)
                        results = list(ddgs.text(keywords=question, region="cn-zh",
                                                  max_results=max_results))
                if results:
                    sources = []
                    for r in results:
                        sources.append({
                            "document": r.get("title", "网络来源"),
                            "snippet": r.get("body", ""),
                            "href": r.get("href", ""),
                            "score": 0.0,
                            "type": "web",
                        })
                    logger.info(f"网络搜索返回 {len(sources)} 条结果 (pkg={pkg_name}, backend={backend})")
                    return sources
            except Exception as e:
                logger.warning(f"搜索失败 (pkg={pkg_name}, backend={backend}): {e}")
                continue

    logger.warning("所有搜索后端均失败")
    return []


# ══════════════════════════════════════════════════
# LLM 生成
# ══════════════════════════════════════════════════

PROMPT_TEMPLATE = """你是一个校园规章制度智能咨询助手。请根据以下【参考资料】回答用户问题。

要求：
1. 从参考资料中提取与问题直接相关的内容，整理成有条理的回答
2. 如果参考资料中包含相关规定的具体条款，请逐条列出关键内容
3. 如果资料中确实没有相关信息，请明确说明，不要编造
4. 回答要准确、简洁、通俗易懂，用编号列表呈现
5. 如果参考资料来自网络搜索，请在回答末尾标注"以上信息来源于网络，仅供参考"

【用户问题】
{question}

【参考资料】
{context}

请回答："""

PROMPT_TEMPLATE_WEB = """你是一个校园规章制度智能咨询助手。知识库中没有找到与用户问题直接相关的信息，以下是从网络搜索到的参考内容。

要求：
1. 基于网络搜索结果，整理出与问题相关的回答
2. 回答要简洁、通俗易懂
3. 必须在回答末尾标注"以上信息来源于网络，仅供参考"
4. 如果网络搜索结果也没有相关信息，请如实说明

【用户问题】
{question}

【网络搜索结果】
{context}

请回答："""


def generate_with_llm(question: str, context_texts: List[str], llm,
                       prompt_template: str = None) -> str:
    """调用 LLM 生成最终回答"""
    from langchain_core.messages import HumanMessage

    if prompt_template is None:
        prompt_template = PROMPT_TEMPLATE

    context_str = "\n\n---\n\n".join(
        f"[资料{i+1}] {text}" for i, text in enumerate(context_texts)
    )
    prompt = prompt_template.format(question=question, context=context_str)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        return _format_fallback_answer(context_texts)


def _format_fallback_answer(contexts: List[str]) -> str:
    """当 LLM 不可用时，格式化检索结果作为回退回答"""
    answer = "以下是从知识库中找到的相关规定：\n\n"
    for i, ctx in enumerate(contexts):
        snippet = ctx[:300] + ("..." if len(ctx) > 300 else "")
        answer += f"{i+1}. {snippet}\n\n"
    answer += "\n（注：当前系统处于仅检索模式，未配置大语言模型）"
    return answer


# ══════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════

def query_rag(question: str, llm=None, collection=None,
              vectorizer=None, top_k: int = 5) -> Dict[str, Any]:
    """
    RAG 问答主函数
    优先从知识库检索，如果知识库无结果则联网搜索

    Returns:
        {"answer": str, "sources": [...]}
    """
    logger.info(f"收到查询: {question}")

    # 加载向量库（如果未传入）
    need_close = False
    if collection is None:
        collection, vectorizer = load_vector_db()
        need_close = True

    # Step 1: 知识库检索
    sources = query_vector_db(question, collection=collection,
                               vectorizer=vectorizer, top_k=5)

    if sources:
        # 知识库有结果 → 用知识库内容 + LLM 生成
        context_texts = [s["snippet"] for s in sources]

        if llm is not None:
            answer = generate_with_llm(question, context_texts, llm, PROMPT_TEMPLATE)
            
            # 检查 LLM 是否判断资料不相关（回答中包含否定关键词）
            no_info_keywords = ["没有找到", "没有相关信息", "资料中不", "资料中没有",
                                "无法从现有资料", "未包含", "与问题无关", "无关"]
            need_web = any(kw in answer for kw in no_info_keywords)
            
            if need_web:
                logger.info("LLM 判断知识库资料不相关，尝试联网搜索...")
                web_sources = web_search(question, max_results=3)
                if web_sources:
                    web_contexts = [f"{s['document']}: {s['snippet']}" for s in web_sources]
                    web_answer = generate_with_llm(question, web_contexts, llm, PROMPT_TEMPLATE_WEB)
                    # 合并来源
                    sources.extend(web_sources)
                    answer = web_answer
                    logger.info(f"联网搜索补充回答完成")
        else:
            answer = _format_fallback_answer(context_texts)

        logger.info(f"回答长度: {len(answer)} 字符")
        return {"answer": answer, "sources": sources}

    # Step 2: 知识库无结果 → 联网搜索
    logger.info("知识库无结果，尝试联网搜索...")
    web_sources = web_search(question, max_results=3)

    if web_sources and llm is not None:
        # 有网络结果 + LLM → 用 LLM 整理网络搜索结果
        context_texts = [f"{s['document']}: {s['snippet']}" for s in web_sources]
        answer = generate_with_llm(question, context_texts, llm, PROMPT_TEMPLATE_WEB)
        logger.info(f"网络搜索回答长度: {len(answer)} 字符")
        return {"answer": answer, "sources": web_sources}

    if web_sources:
        # 有网络结果但无 LLM → 简单罗列
        answer = "知识库中未找到相关信息，以下来自网络搜索：\n\n"
        for i, s in enumerate(web_sources):
            answer += f"{i+1}. {s['document']}\n   {s['snippet']}\n\n"
        answer += "\n（以上信息来源于网络，仅供参考）"
        return {"answer": answer, "sources": web_sources}

    # 知识库和网络都没有结果
    return {
        "answer": "很抱歉，在知识库和网络中都没有找到与您问题相关的信息。",
        "sources": [],
    }
