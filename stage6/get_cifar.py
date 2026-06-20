"""
阶段⑥ / 数据下载：多源轮询 + 断点续传，把 CIFAR-10 弄到本地。

背景：torchvision 默认从 toronto.edu 下 CIFAR-10，但国内常常连不上(502/超时)。
本脚本换用国内可达的镜像（fast.ai S3 等），断点续传、反复重试，直到下完并解压。
下载的是图片文件夹格式（cifar10/train/<类>/*.png, cifar10/test/...），
train_vit.py 会用 torchvision 的 ImageFolder 直接加载。

可后台运行：  python stage6/get_cifar.py
下完会在 stage6/data/cifar10/ 下得到 train/ 和 test/ 两个文件夹。
"""

import os
import sys
import time
import tarfile
import subprocess
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

# 候选镜像（按可用性排序；fast.ai S3 实测国内可达）
SOURCES = [
    ("https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz", "cifar10.tgz"),
    ("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "cifar-10-python.tar.gz"),
]

EXTRACTED_OK = os.path.join(DATA, "cifar10", "train")   # 解压成功的标志目录


def remote_size(url):
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=20) as r:
            return int(r.headers.get("Content-Length", 0))
    except Exception:
        return 0


def download_resume(url, dest):
    """用 curl 断点续传下载一次（一轮）。返回本地文件当前大小。"""
    # -L 跟随重定向, -C - 断点续传, --max-time 单轮最长 10 分钟
    cmd = ["curl", "-L", "-C", "-", "--max-time", "600",
           "--retry", "3", "--retry-delay", "5",
           "-o", dest, url]
    subprocess.run(cmd, cwd=DATA)
    return os.path.getsize(dest) if os.path.exists(dest) else 0


def try_extract(path):
    """尝试解压 tgz/tar.gz 到 DATA。成功且出现 train 目录则返回 True。"""
    try:
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(DATA)
        return True
    except Exception as e:
        print(f"  解压失败（可能没下完）：{e}")
        return False


def already_done():
    if os.path.isdir(EXTRACTED_OK):
        n = sum(len(files) for _, _, files in os.walk(EXTRACTED_OK))
        if n > 1000:   # CIFAR-10 训练集 5 万张，几千张以上就算解压成功
            return True
    return False


def main():
    if already_done():
        print("CIFAR-10 已就绪，无需下载。")
        return

    round_no = 0
    while not already_done():
        round_no += 1
        print(f"\n===== 第 {round_no} 轮下载尝试 =====")
        for url, fname in SOURCES:
            dest = os.path.join(DATA, fname)
            total = remote_size(url)
            have = os.path.getsize(dest) if os.path.exists(dest) else 0
            print(f"[{fname}] 远端 {total/1e6:.1f}MB，本地已有 {have/1e6:.1f}MB，续传中 ...")
            try:
                size = download_resume(url, dest)
            except FileNotFoundError:
                print("  curl 不可用，跳过该源")
                continue
            print(f"  本轮结束，本地 {size/1e6:.1f}MB")

            # 下完整了就解压
            if total > 0 and size >= total:
                print("  下载完整，开始解压 ...")
                if try_extract(dest):
                    if already_done():
                        print(f"\n✅ CIFAR-10 准备完成：{EXTRACTED_OK}")
                        return
            elif total == 0 and size > 100 * 1e6:
                # 拿不到总大小但文件已经很大，也试着解压看看
                if try_extract(dest) and already_done():
                    print(f"\n✅ CIFAR-10 准备完成：{EXTRACTED_OK}")
                    return

        if not already_done():
            print("本轮未完成，10 秒后重试（断点续传会接着下）...")
            time.sleep(10)


if __name__ == "__main__":
    main()
