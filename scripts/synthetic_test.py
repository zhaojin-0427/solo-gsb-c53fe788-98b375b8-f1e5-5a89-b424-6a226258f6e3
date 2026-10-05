"""合成场景端到端测试：验证去重影流水线的关键行为。

场景：亮天空（参照帧中过曝）+ 暗地面 + 一辆运动的深色汽车。
- 汽车在帧 0、1（短曝光）位于 P1，在参照帧 2 位于 P2。
- 期望：P1 处参照帧过曝，虽然两张非参照帧都在 P1 拍到了汽车，
  但 P1 不得保留汽车（应输出过曝天空并标记不确定）；
  P2 处保留参照帧中的汽车；静态区域正常恢复。
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from pipeline import PipelineError, run_pipeline  # noqa: E402

H, W = 480, 640
P1 = (100, 80)    # 汽车在帧 0、1 中的位置 (x, y)
P2 = (300, 80)    # 汽车在参照帧中的位置
CAR_W, CAR_H = 60, 40
SHUTTERS = [1 / 500, 1 / 250, 1 / 60]
REF = 2
SHIFTS = [(2, -1), (-2, 2), (0, 0)]

failures = []


def check(cond, msg):
    print(("  ✔ " if cond else "  ✘ ") + msg)
    if not cond:
        failures.append(msg)


def build_scene():
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    horizon = H * 0.6
    sky = 120.0 - 95.0 * (yy / horizon)          # 120 -> 25 的亮天空
    ground = 2.0 + 0.6 * np.sin(xx / 17.0) + 0.3 * np.cos(yy / 11.0)
    scene = np.where(yy < horizon, sky, ground).astype(np.float32)
    # 静态纹理，便于特征配准
    for _ in range(40):
        x, y = int(rng.integers(0, W - 30)), int(rng.integers(0, H - 20))
        r = int(rng.integers(4, 14))
        v = float(rng.uniform(0.5, 8.0))
        cv2.circle(scene, (x, y), r, v, -1)
    scene += rng.uniform(0, 0.05, scene.shape).astype(np.float32)
    b = scene * 0.95
    g = scene
    r_ = scene * 1.05
    return np.stack([b, g, r_], axis=-1).astype(np.float32)  # BGR


def render(scene, t, car_pos, shift):
    img = scene.copy()
    if car_pos is not None:
        x, y = car_pos
        cv2.rectangle(img, (x, y), (x + CAR_W, y + CAR_H),
                      (1.0, 0.9, 0.8), -1)  # 深色汽车（辐照度约 1）
    z = np.clip(img * t, 0.0, 1.0) ** (1 / 2.2)
    u8 = np.round(z * 255).astype(np.uint8)
    m = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    u8 = cv2.warpAffine(u8, m, (W, H), borderValue=(0, 0, 0))
    # 走一遍 JPEG 编解码，贴近真实输入
    buf = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, 92])[1]
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def region_mean(img, pos, pad=8):
    x, y = pos
    return img[y + pad:y + CAR_H - pad, x + pad:x + CAR_W - pad].mean()


def main():
    scene = build_scene()
    cars = [P1, P1, P2]  # 汽车在两张非参照帧位于旧位置 P1，参照帧位于 P2
    images = [render(scene, SHUTTERS[i], cars[i], SHIFTS[i]) for i in range(3)]

    ldr, mask, ref, stats = run_pipeline(images, SHUTTERS, REF)
    print("stats:", stats)

    p1_result = region_mean(ldr, P1)
    p1_mask = region_mean(mask, P1)
    p2_result = region_mean(ldr, P2)
    p2_mask = region_mean(mask, P2)
    ground_mask = mask[420:460, 380:620].mean()
    sky_static_mask = mask[24:56, 460:580].mean()   # 静态过曝天空（无车、远离边缘）
    print(f"P1(旧位置): result={p1_result:.1f} mask={p1_mask:.1f}")
    print(f"P2(参照位置): result={p2_result:.1f} mask={p2_mask:.1f}")
    print(f"地面(静态): mask={ground_mask:.1f}  静态过曝天空: mask={sky_static_mask:.1f}")

    # 1) 旧位置 P1：参照帧过曝，不得把汽车当高光细节留下 -> 输出应明显亮于汽车
    check(p1_result > 120 and p1_result > p2_result + 60,
          f"P1 不保留旧汽车（result={p1_result:.1f}，汽车={p2_result:.1f}）")
    # 2) P1 无法可靠恢复 -> 必须标记不确定
    check(p1_mask > 200, f"P1 标记为不确定（mask={p1_mask:.1f}）")
    # 3) 参照位置 P2：汽车应保留（暗）
    check(p2_result < 120, f"P2 保留参照帧中的汽车（result={p2_result:.1f}）")
    # 4) P2 来自参照帧有效内容 -> 确定
    check(p2_mask < 55, f"P2 不标记不确定（mask={p2_mask:.1f}）")
    # 5) 静态地面不应被误判为运动/不确定
    check(ground_mask < 5, f"静态地面确定（mask={ground_mask:.1f}）")
    # 6) 静态过曝天空应被短曝光帧恢复，而不是全部标不确定
    check(sky_static_mask < 5, f"静态过曝天空已恢复（mask={sky_static_mask:.1f}）")
    # 7) 确实检测到了运动区域
    check(stats["motion_pct"] > 0.5, f"运动区域占比 {stats['motion_pct']}%")

    # 8) 重叠不足应拒绝输出
    shifted = cv2.warpAffine(
        images[1], np.float32([[1, 0, W * 0.7], [0, 1, 0]]), (W, H),
        borderValue=(0, 0, 0))
    try:
        run_pipeline([images[0], shifted, images[2]], SHUTTERS, REF)
        check(False, "重叠不足应抛出 PipelineError")
    except PipelineError as e:
        check(True, f"重叠不足正确报错（{e}）")

    if failures:
        print(f"\n{len(failures)} 条断言失败")
        sys.exit(1)
    print("\n全部断言通过 ✔")


if __name__ == "__main__":
    main()
