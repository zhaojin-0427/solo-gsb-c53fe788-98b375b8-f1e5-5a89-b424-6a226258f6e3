"""FastAPI 服务：HDR 曝光包围去重影合成台。"""
from __future__ import annotations

import json
import math
import os
import re
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pipeline import PipelineError, run_pipeline

app = FastAPI(title="HDR 曝光包围去重影合成台")

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_BYTES = 40 * 1024 * 1024  # 单张图片上限 40MB
JOB_RE = re.compile(r"^[0-9a-f]{12}$")
RESULT_FILES = {"result.png", "mask.png", "reference.png", "meta.json"}


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/merge")
def merge(
    files: list[UploadFile] = File(...),
    shutters: str = Form(...),
    reference: int = Form(...),
):
    # ---- 数量校验 ----
    if not 3 <= len(files) <= 7:
        raise HTTPException(400, f"需要上传 3～7 张 JPEG 图片，当前为 {len(files)} 张")

    # ---- 快门校验：有限正数（秒）----
    try:
        shutter_list = [float(x) for x in json.loads(shutters)]
    except (TypeError, ValueError, json.JSONDecodeError):
        raise HTTPException(
            400, "快门时间格式非法：应为 JSON 数组，例如 [0.004, 0.016, 0.0625]")
    if len(shutter_list) != len(files):
        raise HTTPException(400, "快门时间数量与图片数量不一致")
    for s in shutter_list:
        if not (math.isfinite(s) and s > 0):
            raise HTTPException(400, f"非法快门时间 {s}：必须为有限正数（单位：秒）")

    if not 0 <= reference < len(files):
        raise HTTPException(400, "参照帧索引超出范围")

    # ---- 解码校验：JPEG 魔数 + 可解码 ----
    images = []
    for idx, f in enumerate(files):
        data = f.file.read()
        if not data:
            raise HTTPException(400, f"第 {idx + 1} 张图片为空")
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(400, f"第 {idx + 1} 张图片超过 40MB 上限")
        if data[:2] != b"\xff\xd8":
            raise HTTPException(400, f"第 {idx + 1} 张不是 JPEG 图片")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, f"第 {idx + 1} 张图片损坏，无法解码")
        images.append(img)

    # ---- 尺寸一致性 ----
    if len({im.shape[:2] for im in images}) > 1:
        raise HTTPException(400, "图片尺寸不一致：所有图片必须具有相同的宽和高")

    # ---- 合成 ----
    try:
        ldr, mask, ref_img, stats = run_pipeline(images, shutter_list, reference)
    except PipelineError as e:
        raise HTTPException(400, str(e))

    # ---- 保存结果 ----
    job = uuid.uuid4().hex[:12]
    out = RESULTS_DIR / job
    out.mkdir(parents=True)
    cv2.imwrite(str(out / "result.png"), ldr)
    cv2.imwrite(str(out / "mask.png"), mask)
    cv2.imwrite(str(out / "reference.png"), ref_img)
    (out / "meta.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "job_id": job,
        "result_url": f"/api/results/{job}/result.png",
        "mask_url": f"/api/results/{job}/mask.png",
        "reference_url": f"/api/results/{job}/reference.png",
        "stats": stats,
    }


@app.get("/api/results/{job}/{name}")
def get_result(job: str, name: str):
    if not JOB_RE.match(job) or name not in RESULT_FILES:
        raise HTTPException(404, "结果不存在")
    path = RESULTS_DIR / job / name
    if not path.exists():
        raise HTTPException(404, "结果不存在")
    media = "application/json" if name.endswith(".json") else "image/png"
    return FileResponse(path, media_type=media)


# 前端静态页面（挂在最后，避免覆盖 /api 路由）
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True),
          name="static")
