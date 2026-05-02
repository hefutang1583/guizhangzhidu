"""
重建向量库并测试完整流程
解决Python 3.14 + ChromaDB兼容性问题
"""
import sys
import os
import shutil

sys.stdout.reconfigure(encoding='utf-8')
os.chdir(r'c:\Users\Administrator\Desktop\MyRAGProject')

# Step 1: 删除旧的向量库
print("=" * 50)
print("Step 1: 清理旧向量库")
print("=" * 50, flush=True)
if os.path.exists('./vector_db'):
    shutil.rmtree('./vector_db')
    print("   已删除旧的 vector_db 目录", flush=True)
else:
    print("   无需清理", flush=True)

# Step 2: 导入并构建
print("\n" + "=" * 50)
print("Step 2: 构建新向量库")
print("=" * 50, flush=True)

os.environ['EMBED_MODE'] = 'tfidf'
os.environ['OPENAI_API_KEY'] = ''

from build_knowledge_base import build_knowledge_base

result = build_knowledge_base()
print(f"   结果: {result}", flush=True)

# Step 3: 验证
print("\n" + "=" * 50)
print("Step 3: 验证向量库")
print("=" * 50, flush=True)

import chromadb
client = chromadb.PersistentClient(path='./vector_db')
cols = client.list_collections()
print(f"   集合数: {len(cols)}", flush=True)

for col in cols:
    name = col.name
    # 尝试获取数据（避免调用count）
    try:
        r = col.get(limit=2, include=['documents', 'metadatas'])
        n_docs = len(r['ids'])
        sources = set(m.get('source','') for m in r['metadatas'])
        print(f"   集合 '{name}': {n_docs} 条样本记录", flush=True)
        for i, (doc, meta) in enumerate(zip(r['documents'], r['metadatas'])):
            src = meta.get('source', 'N/A')[:40]
            snippet = doc[:60].replace('\n', ' ')
            print(f"     [{i+1}] {src}: {snippet}...", flush=True)
        
        # 测试查询
        from build_knowledge_base import TfidfEmbeddingFunction
        embed_fn = TfidfEmbeddingFunction()
        qr = col.query(query_embeddings=embed_fn(["学生请假"])[0], n_results=2)
        print(f"   检索测试 '学生请假': 返回 {len(qr['ids'][0])} 条结果", flush=True)
        for j, m in enumerate(qr['metadatas'][0]):
            print(f"     -> {m.get('source', '')[:40]}", flush=True)
            
    except Exception as e:
        print(f"   错误: {type(e).__name__}: {e}", flush=True)

# Step 4: FastAPI测试
print("\n" + "=" * 50)
print("Step 4: FastAPI 加载测试")
print("=" * 50, flush=True)

try:
    from main import app
    routes = [f"{list(r.methods)} {r.path}" for r in app.routes if hasattr(r,'path') and hasattr(r,'methods')]
    print(f"   FastAPI 加载成功! 共 {len(routes)} 个路由:", flush=True)
    for r in routes[:6]:
        print(f"     {r}", flush=True)
except Exception as e:
    print(f"   错误: {e}", flush=True)

print("\n" + "=" * 50)
print("所有步骤完成! 系统就绪。")
print("启动命令: python main.py")
print("=" * 50, flush=True)
