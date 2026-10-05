"""生成一组合成包围曝光演示图片（含运动物体与轻微错位），用于快速体验合成台。

用法：
    python3 scripts/make_demo.py [输出目录]     # 默认 ./demo

生成 4 张 JPEG（frame0..frame3.jpg）并打印应填写的快门时间与建议参照帧。
依赖：numpy、opencv-python（或 opencv-python-headless）。
"""
import os
import sys

import cv2
import numpy as np

H, W = 480, 640
SHUTTERS = [1 / 500, 1 / 250, 1 / 60, 1 / 15]   # 秒
REF = 2                                          # 建议参照帧下标
CAR_W, CAR_H = 60, 40
# 汽车在各帧中的位置（在参照帧中位于画面中部）
CAR_POS = [(90, 80), (140, 80), (300, 80), (420, 80)]
SHIFTS = [(2, -1), (-2, 2), (0, 0), (3, 1)]      # 模拟手持轻微错位


def build_scene():
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    horizon = H * 0.6
    sky = 120.0 - 95.0 * (yy / horizon)          # 亮天空：参照帧中过曝
    ground = 2.0 + 0.6 * np.sin(xx / 17.0) + 0.3 * np.cos(yy / 11.0)
    scene = np.where(yy < horizon, sky, ground).astype(np.float32)
    for _ in range(40):                          # 静态纹理，便于配准
        x, y = int(rng.integers(0, W - 30)), int(rng.integers(0, H - 20))
        r = int(rng.integers(4, 14))
        v = float(rng.uniform(0.5, 8.0))
        cv2.circle(scene, (x, y), r, v, -1)
    scene += rng.uniform(0, 0.05, scene.shape).astype(np.float32)
    return np.stack([scene * 0.95, scene, scene * 1.05], axis=-1).astype(np.float32)


def render(scene, t, car_pos, shift):
    img = scene.copy()
    x, y = car_pos
    cv2.rectangle(img, (x, y), (x + CAR_W, y + CAR_H), (1.0, 0.9, 0.8), -1)
    z = np.clip(img * t, 0.0, 1.0) ** (1 / 2.2)
    u8 = np.round(z * 255).astype(np.uint8)
    m = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    return cv2.warpAffine(u8, m, (W, H), borderValue=(0, 0, 0))


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "demo"
    os.makedirs(out_dir, exist_ok=True)
    scene = build_scene()
    for i in range(len(SHUTTERS)):
        img = render(scene, SHUTTERS[i], CAR_POS[i], SHIFTS[i])
        path = os.path.join(out_dir, f"frame{i}.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"已生成 {path}")
    print("\n在页面中填写：")
    for i, t in enumerate(SHUTTERS):
        print(f"  frame{i}.jpg  快门 = {t:.6f} 秒（即 1/{round(1/t)}）")
    print(f"  参照帧选择第 {REF + 1} 张（frame{REF}.jpg）")


if __name__ == "__main__":
    main()
