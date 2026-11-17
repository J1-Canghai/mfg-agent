"""Manual RAG: embed the question, take the top-k chunks by cosine similarity,
and let the chat model answer with citations.

Usage: python ingest.py, then python rag.py
"""
import json
from pathlib import Path

import numpy as np

from config import CHAT_MODEL, EMBED_MODEL, chat_client, embed_client

BASE = Path(__file__).parent
CHUNK_FILE = BASE / "data" / "chunks.jsonl"
VECTOR_FILE = BASE / "data" / "chunk_vectors.npy"

BATCH_SIZE = 10  # API limit for text-embedding-v4

SYSTEM_PROMPT = """你是工厂设备维护助手，只能根据【资料】回答问题。
要求：
1. 每个结论后面用 [编号] 标出出自哪条资料，例如 [2]
2. 资料里没有的内容，直接回答"手册中没有找到相关信息"，不要自己编
3. 涉及安全的步骤要完整写出，不要省略
4. 回答简洁，分点写"""


def embed(texts):
    """Return L2-normalized embeddings, so a dot product equals cosine similarity."""
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        resp = embed_client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + BATCH_SIZE])
        vectors.extend(item.embedding for item in resp.data)
    V = np.array(vectors, dtype=np.float32)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def load_chunks():
    with CHUNK_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class ManualIndex:
    def __init__(self):
        self.chunks = load_chunks()
        # reuse cached vectors unless chunks.jsonl was rebuilt after them
        cache_ok = (
            VECTOR_FILE.exists()
            and VECTOR_FILE.stat().st_mtime > CHUNK_FILE.stat().st_mtime
        )
        if cache_ok:
            self.vectors = np.load(VECTOR_FILE)
        if not cache_ok or len(self.vectors) != len(self.chunks):
            print(f"Embedding {len(self.chunks)} chunks...")
            self.vectors = embed([c["text"] for c in self.chunks])
            np.save(VECTOR_FILE, self.vectors)

    def search(self, query, k=4):
        q = embed([query])[0]
        scores = self.vectors @ q
        top = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


_index = None


def get_index():
    # built lazily so importing this module doesn't call the API
    global _index
    if _index is None:
        _index = ManualIndex()
    return _index


def format_context(hits):
    blocks = []
    for n, h in enumerate(hits, start=1):
        blocks.append(f"[{n}] 来源：{h['source']} · {h['section']}\n{h['text']}")
    return "\n\n".join(blocks)


def answer(question, k=4):
    hits = get_index().search(question, k)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【资料】\n{format_context(hits)}\n\n【问题】{question}"},
    ]
    resp = chat_client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
    return resp.choices[0].message.content, hits


def main():
    print("Manual Q&A (q to quit)")
    while True:
        question = input("\nQ: ").strip()
        if question.lower() == "q":
            break
        if not question:
            continue
        text, hits = answer(question)
        print(f"\nA: {text}\n\nSources:")
        for n, h in enumerate(hits, start=1):
            print(f"  [{n}] {h['source']} · {h['section']} (score {h['score']:.3f})")


if __name__ == "__main__":
    main()
