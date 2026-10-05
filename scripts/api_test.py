"""API 层测试：正常合成流程 + 非法快门 / 损坏图片 / 尺寸不一致 / 数量错误。"""
import io
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
os.environ.setdefault("RESULTS_DIR", "/tmp/hdr_test_results")

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from synthetic_test import build_scene, render, SHUTTERS, P1, P2, SHIFTS  # noqa: E402

client = TestClient(app)
failures = []


def check(cond, msg):
    print(("  ✔ " if cond else "  ✘ ") + msg)
    if not cond:
        failures.append(msg)


def jpeg_bytes(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def make_files(images):
    return [("files", (f"frame{i}.jpg", jpeg_bytes(im), "image/jpeg"))
            for i, im in enumerate(images)]


def post(images, shutters, reference=2):
    return client.post(
        "/api/merge",
        files=make_files(images),
        data={"shutters": json.dumps(shutters), "reference": str(reference)},
    )


def main():
    scene = build_scene()
    cars = [P1, P1, P2]
    images = [render(scene, SHUTTERS[i], cars[i], SHIFTS[i]) for i in range(3)]

    # ---- 正常流程 ----
    r = post(images, SHUTTERS)
    check(r.status_code == 200, f"正常合成返回 200（实际 {r.status_code}: {r.text[:200]}）")
    if r.status_code == 200:
        d = r.json()
        for key in ("result_url", "mask_url", "reference_url", "stats"):
            check(key in d, f"响应包含 {key}")
        for key in ("result_url", "mask_url", "reference_url"):
            rr = client.get(d[key])
            check(rr.status_code == 200 and rr.content[:4] == b"\x89PNG",
                  f"可下载 {key}（PNG）")
        check(d["stats"]["motion_pct"] > 0, "统计包含运动区域占比")

    # ---- 非法快门 ----
    r = post(images, [0.004, -0.016, 0.0625])
    check(r.status_code == 400 and "快门" in r.text, f"负快门报错（{r.status_code}）")
    r = post(images, [0.004, float("inf"), 0.0625])
    check(r.status_code == 400, f"无穷快门报错（{r.status_code}）")
    r = client.post("/api/merge", files=make_files(images),
                    data={"shutters": "not-json", "reference": "2"})
    check(r.status_code == 400, f"快门格式非法报错（{r.status_code}）")
    r = post(images, [0.004, 0.016])
    check(r.status_code == 400 and "数量" in r.text, f"快门数量不符报错（{r.status_code}）")

    # ---- 损坏图片 ----
    bad = images.copy()
    r = client.post(
        "/api/merge",
        files=[("files", ("a.jpg", jpeg_bytes(images[0]), "image/jpeg")),
               ("files", ("b.jpg", b"\xff\xd8\xff\xd9garbagebytes", "image/jpeg")),
               ("files", ("c.jpg", jpeg_bytes(images[2]), "image/jpeg"))],
        data={"shutters": json.dumps(SHUTTERS), "reference": "2"})
    check(r.status_code == 400 and "损坏" in r.text, f"损坏 JPEG 报错（{r.status_code}）")
    r = client.post(
        "/api/merge",
        files=[("files", ("a.jpg", jpeg_bytes(images[0]), "image/jpeg")),
               ("files", ("b.png", b"\x89PNG\r\n\x1a\n....", "image/png")),
               ("files", ("c.jpg", jpeg_bytes(images[2]), "image/jpeg"))],
        data={"shutters": json.dumps(SHUTTERS), "reference": "2"})
    check(r.status_code == 400 and "JPEG" in r.text, f"非 JPEG 报错（{r.status_code}）")

    # ---- 尺寸不一致 ----
    small = cv2.resize(images[1], (320, 240))
    r = post([images[0], small, images[2]], SHUTTERS)
    check(r.status_code == 400 and "尺寸" in r.text, f"尺寸不一致报错（{r.status_code}）")

    # ---- 数量错误 ----
    r = post(images[:2], SHUTTERS[:2], reference=1)
    check(r.status_code == 400 and "3～7" in r.text, f"图片过少报错（{r.status_code}）")

    # ---- 参照帧索引越界 ----
    r = post(images, SHUTTERS, reference=5)
    check(r.status_code == 400 and "参照" in r.text, f"参照帧越界报错（{r.status_code}）")

    # ---- 静态资源 ----
    r = client.get("/")
    check(r.status_code == 200 and "合成" in r.text, "首页可访问")
    r = client.get("/api/health")
    check(r.status_code == 200 and r.json().get("ok"), "健康检查通过")

    if failures:
        print(f"\n{len(failures)} 条断言失败")
        sys.exit(1)
    print("\nAPI 测试全部通过 ✔")


if __name__ == "__main__":
    main()
