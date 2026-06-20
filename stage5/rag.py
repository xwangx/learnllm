"""
阶段⑤ / RAG 问答：检索 wiki 相关原文 + 让 0.5B 基于原文作答 + 列出处。

每次提问的流程：
  1. 把问题嵌入成向量
  2. 和索引里 9000+ 个块向量算余弦相似度，取最相关的 top-k 段
  3. 把这几段原文拼进提示，喂给 Qwen2.5-0.5B 生成答案
  4. 答案 + 列出每段来源（文件 / 小节 / 相似度）

和阶段④微调的本质区别：答案**基于检索到的真实原文**，可溯源；wiki 更新只需重建索引。

用法：
  python stage5/rag.py --q "什么是数据血缘？"
  python stage5/rag.py                 # 交互模式
  python stage5/rag.py --k 6           # 检索更多段
  python stage5/rag.py --show          # 同时打印检索到的原文
"""

import os
import json
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_index import embed_texts, INDEX_PATH, META_PATH

HERE = os.path.dirname(os.path.abspath(__file__))
GEN_MODEL_DIR = os.path.join(HERE, "..", "stage4", "Qwen2.5-0.5B")
device = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM = ("你是数据治理领域的助手。请严格根据【参考资料】回答【问题】，"
          "不要编造资料里没有的内容；若资料未提及，就回答“参考资料中未提到”。")


def load_index():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        raise SystemExit("索引不存在，请先运行: python stage5/build_index.py")
    embeddings = np.load(INDEX_PATH)["embeddings"]      # (N, dim), 已归一化
    meta = json.load(open(META_PATH, encoding="utf-8"))
    return embeddings, meta


def retrieve(query, embeddings, meta, k=4):
    """检索 top-k 最相关的块。返回 [(score, chunk), ...]。"""
    q = embed_texts([query], is_query=True)[0]          # (dim,), 已归一化
    scores = embeddings @ q                              # 点积=余弦相似度 (N,)
    idx = np.argsort(-scores)[:k]
    return [(float(scores[i]), meta[i]) for i in idx]


def build_prompt(query, hits):
    refs = []
    for i, (score, c) in enumerate(hits, 1):
        refs.append(f"[{i}] （来自《{c['title']}》- {c['section']}）\n{c['text']}")
    refs_text = "\n\n".join(refs)
    return (f"【参考资料】\n{refs_text}\n\n【问题】{query}\n\n"
            f"请根据上述参考资料用中文简洁回答。")


def generate(model, tokenizer, prompt, max_new_tokens=300):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, repetition_penalty=1.1,
                             pad_token_id=tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def answer_one(query, model, tokenizer, embeddings, meta, k, show):
    hits = retrieve(query, embeddings, meta, k)
    if show:
        print("\n--- 检索到的原文 ---")
        for i, (score, c) in enumerate(hits, 1):
            print(f"[{i}] ({score:.3f}) 《{c['title']}》- {c['section']}")
            print(f"    {c['text'][:120].replace(chr(10),' ')}...")
    prompt = build_prompt(query, hits)
    ans = generate(model, tokenizer, prompt)
    print("\n💬 答案：")
    print(ans)
    print("\n📚 出处：")
    for i, (score, c) in enumerate(hits, 1):
        print(f"  [{i}] 《{c['title']}》- {c['section']}  (相似度 {score:.3f}, {c['file']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", default=None)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--show", action="store_true", help="打印检索到的原文")
    args = parser.parse_args()

    print("加载索引和生成模型 ...")
    embeddings, meta = load_index()
    model_path = GEN_MODEL_DIR if os.path.isdir(GEN_MODEL_DIR) else "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16).to(device).eval()
    print(f"索引 {embeddings.shape[0]} 块，就绪。\n")

    if args.q is not None:
        answer_one(args.q, model, tokenizer, embeddings, meta, args.k, args.show)
        return

    print("进入交互模式（输入 q 退出）")
    while True:
        try:
            q = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("q", "quit", "exit"):
            break
        if q:
            answer_one(q, model, tokenizer, embeddings, meta, args.k, args.show)


if __name__ == "__main__":
    main()
