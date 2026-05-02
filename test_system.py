"""快速验证向量库和API状态 - 结果写入JSON"""
import sys
import os
import json

os.chdir(r'c:\Users\Administrator\Desktop\MyRAGProject')

results = {"status": "ok", "checks": []}

# 1. 检查向量库
try:
    import chromadb
    client = chromadb.PersistentClient(path='./vector_db')
    col = client.get_collection('campus_rules')
    count = col.count()
    
    check1 = {
        "name": "向量库检查",
        "status": "pass",
        "records": count,
    }
    
    if count > 0:
        sample = col.get(limit=1, include=['documents', 'metadatas'])
        check1["sample_source"] = sample['metadatas'][0].get('source', 'N/A')
        check1["sample_content"] = sample['documents'][0][:100]
        
        # 测试检索
        from build_knowledge_base import TfidfEmbeddingFunction
        embed_fn = TfidfEmbeddingFunction()
        query = "学生请假制度"
        q_results = col.query(query_embeddings=embed_fn([query])[0], n_results=3)
        check1["query_test"] = query
        check1["retrieved"] = len(q_results['documents'][0])
        check1["top_sources"] = [m.get('source','') for m in q_results['metadatas'][0]]
    else:
        check1["status"] = "empty"
    
    results["checks"].append(check1)
except Exception as e:
    results["checks"].append({"name": "向量库检查", "status": "error", "error": str(e)})

# 2. 测试FastAPI加载
try:
    from main import app
    routes = []
    for r in app.routes:
        if hasattr(r, 'path') and hasattr(r, 'methods'):
            routes.append(f"{r.methods} {r.path}")
    
    results["checks"].append({
        "name": "FastAPI加载",
        "status": "pass",
        "routes": routes,
    })
except Exception as e:
    results["checks"].append({"name": "FastAPI加载", "status": "error", "error": str(e)})

# 写入结果文件
with open('test_result.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("DONE")
