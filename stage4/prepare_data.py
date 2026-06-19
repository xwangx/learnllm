"""
阶段④ / 数据准备：把数据治理 wiki 的 markdown 整理成一份训练语料。

做的事：
  1. 递归遍历 wiki 下所有 .md 文件
  2. 去掉每篇开头的 frontmatter 元信息（--- 之间的部分）
  3. 拼成一份大文本 corpus.txt，供 finetune.py 训练用

运行：  python stage4/prepare_data.py
"""

import os
import glob

# 你的 wiki 路径（如改位置，改这里即可）
WIKI_DIR = r"Z:\work\wiki"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "corpus.txt")


def strip_frontmatter(text):
    """去掉 markdown 开头的 YAML frontmatter（被一对 --- 包起来的元信息）。"""
    if text.startswith("---"):
        end = text.find("\n---", 3)        # 找第二个 ---
        if end != -1:
            # 跳到第二个 --- 那一行之后
            nl = text.find("\n", end + 1)
            if nl != -1:
                return text[nl + 1:]
    return text


def main():
    if not os.path.isdir(WIKI_DIR):
        print(f"找不到 wiki 目录: {WIKI_DIR}")
        return

    md_files = glob.glob(os.path.join(WIKI_DIR, "**", "*.md"), recursive=True)
    print(f"找到 {len(md_files)} 篇 markdown")

    docs = []
    total_chars = 0
    for path in md_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        body = strip_frontmatter(text).strip()
        if len(body) < 50:        # 跳过几乎空白的文件
            continue
        docs.append(body)
        total_chars += len(body)

    # 文档之间用空行隔开，拼成一份大语料
    corpus = "\n\n".join(docs)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(corpus)

    print(f"有效文档 {len(docs)} 篇，正文合计 {total_chars} 字符")
    print(f"已写入 {OUT_PATH} ({len(corpus)} 字符)")
    print("\n--- 语料开头预览 ---")
    print(corpus[:300])


if __name__ == "__main__":
    main()
