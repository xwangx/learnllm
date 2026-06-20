"""
阶段⑤ / 建索引：把 wiki 切块、嵌入成向量、存成索引文件（一次性）。

RAG 的"准备阶段"：
  1. 读 wiki 的 md，按小节切成一块块（每块带来源信息，方便引用出处）
  2. 用 bge 嵌入模型把每块文字变成一个向量（语义相近 -> 向量相近）
  3. 把所有向量 + 元信息存盘，供 rag.py 检索

嵌入函数 load_embedder / embed_texts 会被 rag.py 复用。

运行：  python stage5/build_index.py
"""

import os
import re
import glob
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

WIKI_DIR = r"Z:\work\wiki"
HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(HERE, "index.npz")
META_PATH = os.path.join(HERE, "meta.json")

EMBED_MODEL = "AI-ModelScope/bge-small-zh-v1.5"
# bge 官方建议：查询侧加这个指令前缀，检索效果更好（文档侧不加）
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

MAX_CHUNK = 500     # 每块最多字符数（过长的小节再切）
MIN_CHUNK = 30      # 太短的块丢弃
MIN_CJK = 15        # 至少含这么多汉字（滤掉纯表格/链接/元数据）

JUNK_MARKERS = ["自动生成", "不要手改", "暂无反向引用", "explode_ontology"]
SKIP_SECTIONS = {"source", "referenced in", "properties", "version", "provenance",
                 "changelog", "see also", "references", "backlinks", "metadata",
                 "来源", "反向引用", "属性", "版本"}


# ============================================================================
# 嵌入相关（rag.py 也会 import 这两个函数）
# ============================================================================
_embedder = None


def load_embedder():
    """加载 bge 嵌入模型（单例）。返回 (tokenizer, model, device)。"""
    global _embedder
    if _embedder is None:
        from modelscope import snapshot_download
        path = snapshot_download(EMBED_MODEL)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path).to(device).eval()
        _embedder = (tok, model, device)
    return _embedder


@torch.no_grad()
def embed_texts(texts, is_query=False, batch_size=64):
    """把一批文字编码成归一化向量 (N, dim)。"""
    tok, model, device = load_embedder()
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(device)
        hidden = model(**enc).last_hidden_state      # (B, T, dim)
        # bge 用 [CLS]（第 0 个 token）作为整句表示
        emb = hidden[:, 0]
        emb = F.normalize(emb, p=2, dim=1)           # L2 归一化 -> 之后点积=余弦相似度
        out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0)


# ============================================================================
# 切块
# ============================================================================
def strip_frontmatter(text):
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            nl = text.find("\n", end + 1)
            if nl != -1:
                return text[nl + 1:]
    return text


def count_cjk(s):
    return sum(1 for ch in s if "一" <= ch <= "鿿")


def good_chunk(s):
    s = s.strip()
    return len(s) >= MIN_CHUNK and count_cjk(s) >= MIN_CJK and \
        not any(m in s for m in JUNK_MARKERS)


def split_long(text):
    """把过长文本按段落/长度切成 <= MAX_CHUNK 的小块。"""
    text = text.strip()
    if len(text) <= MAX_CHUNK:
        return [text]
    pieces, buf = [], ""
    for para in text.split("\n"):
        if len(buf) + len(para) + 1 > MAX_CHUNK and buf:
            pieces.append(buf.strip())
            buf = ""
        buf += para + "\n"
    if buf.strip():
        pieces.append(buf.strip())
    return pieces


def parse_doc(text):
    text = strip_frontmatter(text)
    lines = text.split("\n")
    title = None
    for ln in lines:
        if ln.startswith("# "):
            title = ln[2:].strip()
            break
    sections, cur_name, cur_buf, intro, seen = [], None, [], [], False
    for ln in lines:
        if ln.startswith("## "):
            if cur_name is not None:
                sections.append((cur_name, "\n".join(cur_buf).strip()))
            cur_name, cur_buf, seen = ln[3:].strip(), [], True
        elif ln.startswith("# "):
            continue
        else:
            (cur_buf if seen else intro).append(ln)
    if cur_name is not None:
        sections.append((cur_name, "\n".join(cur_buf).strip()))
    return title, "\n".join(intro).strip(), sections


def main():
    md_files = glob.glob(os.path.join(WIKI_DIR, "**", "*.md"), recursive=True)
    print(f"找到 {len(md_files)} 篇 markdown，开始切块 ...")

    chunks = []     # 每个元素: {"file","title","section","text"}
    for path in md_files:
        try:
            text = open(path, "r", encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        title, intro, sections = parse_doc(text)
        if not title:
            continue
        rel = os.path.relpath(path, WIKI_DIR)

        # 文档引言（第一个 ## 之前）
        for piece in split_long(intro):
            if good_chunk(piece):
                chunks.append({"file": rel, "title": title,
                               "section": "(开篇)", "text": piece})
        # 各小节
        for sname, scontent in sections:
            if not sname or sname.lower() in SKIP_SECTIONS:
                continue
            for piece in split_long(scontent):
                if good_chunk(piece):
                    chunks.append({"file": rel, "title": title,
                                   "section": sname, "text": piece})

    print(f"共切出 {len(chunks)} 个文本块，开始嵌入 ...")

    # 嵌入：文档侧不加查询前缀
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, is_query=False)
    print(f"嵌入完成，向量形状 {embeddings.shape}")

    # 存盘：向量存 npz，元信息存 json（顺序对齐）
    np.savez_compressed(INDEX_PATH, embeddings=embeddings.astype(np.float32))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"索引已保存：{INDEX_PATH} + {META_PATH}")


if __name__ == "__main__":
    main()
