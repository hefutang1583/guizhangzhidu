import sys
sys.stdout.reconfigure(encoding='utf-8')

print("A: import", flush=True)
import chromadb
print("B: client", flush=True)
c = chromadb.PersistentClient(path='./vector_db')
print("C: list cols", flush=True)
cols = c.list_collections()
print(f"D: cols={len(cols)}", flush=True)

# 尝试不调用count，直接peek
for col in cols:
    name = col.name
    print(f"E: col={name}", flush=True)
    # 用get代替count
    try:
        r = col.get(limit=1, include=['documents'])
        print(f"F: got {len(r['ids'])} docs", flush=True)
    except Exception as e:
        print(f"F_ERR: {e}", flush=True)

print("G: DONE", flush=True)
