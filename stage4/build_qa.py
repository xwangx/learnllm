"""
阶段④ 指令微调版 / 第一步：从 wiki 结构自动造"问答对"。

指令微调需要 (问题, 答案) 数据，但 wiki 只是文档。这里用规则法：
  · 文章标题 / 小节标题 -> 当问题
  · 对应的正文 / 小节内容 -> 当答案（都是 wiki 的真实文本）
多套问法模板增加多样性。输出 qa.jsonl，每行一条 {"question":..., "answer":...}。

运行：  python stage4/build_qa.py
"""

import os
import re
import glob
import json

WIKI_DIR = r"Z:\work\wiki"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "qa.jsonl")

MIN_ANS, MAX_ANS = 40, 1200      # 答案字符数过滤范围
MIN_CJK = 30                     # 答案至少含这么多汉字（滤掉表格/链接/元数据等"非正文"）

# 答案里出现这些字样 -> 是自动生成/样板内容，丢弃
JUNK_MARKERS = ["自动生成", "不要手改", "暂无反向引用", "explode_ontology",
                "AUTO-GENERATED", "do not edit"]
# 这些小节名是元数据，不是知识正文，跳过
SKIP_SECTIONS = {"source", "referenced in", "properties", "version", "provenance",
                 "changelog", "see also", "references", "backlinks", "metadata",
                 "来源", "反向引用", "属性", "版本"}


def count_cjk(s):
    return sum(1 for ch in s if "一" <= ch <= "鿿")


def is_good_answer(a):
    if not (MIN_ANS <= len(a) <= MAX_ANS):
        return False
    if count_cjk(a) < MIN_CJK:           # 正文太少（多半是表格/链接/元数据）
        return False
    low = a.lower()
    return not any(m.lower() in low for m in JUNK_MARKERS)


def is_good_title(t):
    return "added in vNEXT" not in t and not t.lower().startswith("abandoned")


def clean_title(t):
    # 去掉 "(added in v0.3 from 001, 客户主数据实体)" 这类自动后缀，让问题更自然
    return re.sub(r"\s*\(added in [^)]*\)", "", t).strip()

# 整篇文章的问法模板（{t} = 标题）
DOC_TEMPLATES = ["什么是{t}？", "请介绍一下{t}。", "{t}是指什么？", "请说明{t}的概念。"]
# 小节的问法模板（{t} = 标题, {s} = 小节名）
SEC_TEMPLATES = ["在「{t}」中，{s}是什么？", "关于{t}的{s}，请说明。", "{t}的{s}指什么？"]


def strip_frontmatter(text):
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            nl = text.find("\n", end + 1)
            if nl != -1:
                return text[nl + 1:]
    return text


def clean(s):
    return s.strip()


def parse_doc(text):
    """返回 (标题, 摘要正文, [(小节名, 小节内容), ...])。"""
    text = strip_frontmatter(text)
    lines = text.split("\n")

    # 标题：第一个 "# " 开头的行
    title = None
    for ln in lines:
        if ln.startswith("# "):
            title = clean(ln[2:])
            break

    # 按 "## " 切小节
    sections = []
    cur_name, cur_buf = None, []
    intro_buf = []          # 第一个 ## 之前的正文当摘要
    seen_h2 = False
    for ln in lines:
        if ln.startswith("## "):
            if cur_name is not None:
                sections.append((cur_name, "\n".join(cur_buf).strip()))
            cur_name = clean(ln[3:])
            cur_buf = []
            seen_h2 = True
        elif ln.startswith("# "):
            continue        # 跳过主标题行
        else:
            if seen_h2:
                cur_buf.append(ln)
            else:
                intro_buf.append(ln)
    if cur_name is not None:
        sections.append((cur_name, "\n".join(cur_buf).strip()))

    intro = "\n".join(intro_buf).strip()
    return title, intro, sections


def main():
    md_files = glob.glob(os.path.join(WIKI_DIR, "**", "*.md"), recursive=True)
    print(f"找到 {len(md_files)} 篇 markdown")

    qa_pairs = []
    for path in md_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        title, intro, sections = parse_doc(text)
        if not title or not is_good_title(title):
            continue
        title = clean_title(title)      # 清掉自动后缀，使问题更自然
        if not title:
            continue

        # 整篇：标题 -> 摘要（或第一个小节内容）
        body = intro if len(intro) >= MIN_ANS else (sections[0][1] if sections else "")
        if is_good_answer(body):
            # 每篇只用一个问法模板（用标题长度选，确定可复现），避免答案重复太多次
            tpl = DOC_TEMPLATES[len(title) % len(DOC_TEMPLATES)]
            qa_pairs.append({"question": tpl.format(t=title), "answer": body})

        # 各小节：小节名 -> 小节内容
        for i, (sname, scontent) in enumerate(sections):
            if not sname or sname.lower() in SKIP_SECTIONS:
                continue
            if not is_good_answer(scontent):
                continue
            tpl = SEC_TEMPLATES[(len(sname) + i) % len(SEC_TEMPLATES)]
            qa_pairs.append({"question": tpl.format(t=title, s=sname),
                             "answer": scontent})

    # 写出 jsonl
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"共生成 {len(qa_pairs)} 条问答对，写入 {OUT_PATH}")
    print("\n--- 抽样 3 条 ---")
    for qa in qa_pairs[:3]:
        print(f"\nQ: {qa['question']}")
        print(f"A: {qa['answer'][:120]}...")


if __name__ == "__main__":
    main()
