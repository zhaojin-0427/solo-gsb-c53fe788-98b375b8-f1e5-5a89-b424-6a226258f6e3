"""HDR 曝光包围去重影合成流水线。

流程：
1. 配准：ORB 特征 + RANSAC 估计每张图到参照帧的相似变换（平移/旋转/缩放），
   重投影到参照帧坐标系，并检查公共重叠区域占比。
2. 曝光归一化：固定光圈/ISO/白平衡下，像素线性亮度 / 快门时间 ≈ 场景辐照度。
3. 运动检测：对所有帧两两做一致性检验——
   - 两帧均有效：曝光归一化后差异显著（容忍 ±1px 配准残差与噪声）；
   - 一帧高光裁切（只给出辐照度下界）：另一帧显著更暗 => 场景必然变化；
   - 一帧暗部不可靠（只给出辐照度上界）：另一帧显著更亮 => 场景必然变化。
   因此“参照帧过曝 + 非参照帧在旧位置拍到物体”会被判为运动而不是高光细节；
   纯曝光差异、饱和裁切本身和轻微配准误差不会触发运动。
4. 合成：静态区域对未裁切帧加权平均恢复高光/暗部；运动区域只取参照帧，
   使运动物体只保留在参照帧位置。
5. 不确定区域：所有帧都缺失有效信息、或运动区域中参照帧本身被裁切时，
   保留参照帧内容并在遮罩中标记为不确定。
"""
from __future__ import annotations

import cv2
import numpy as np

SAT_HIGH = 250           # 8bit 高光裁切阈值：任一通道 >= 该值视为饱和
DARK_LOW = 8             # 8bit 暗部不可靠阈值：所有通道 <= 该值视为欠曝
MIN_OVERLAP = 0.5        # 配准后公共重叠区域的最小占比，否则拒绝输出
MAX_PIXELS = 12_000_000  # 处理分辨率上限（超出等比缩小，控制内存与耗时）
ALIGN_MAX_SIDE = 1600    # 特征配准使用的最长边
REL_TOL = 0.20           # 有效像素辐照度相对容差
BOUND_MARGIN = 0.8       # 裁切边界判据的余量系数
GAMMA = 2.2              # JPEG 近似响应曲线

NOCOV, VALID, SAT_HI, SAT_LO = 0, 1, 2, 3
BGR2Y = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)
K3 = np.ones((3, 3), np.uint8)
K5 = np.ones((5, 5), np.uint8)


class PipelineError(Exception):
    """可直接展示给用户的流水线错误（对应 HTTP 400）。"""


def run_pipeline(images, shutters, ref_idx):
    """合成入口。

    images: BGR uint8 列表（已校验同尺寸）；shutters: 各帧快门时间（秒）；
    ref_idx: 参照帧下标。
    返回 (ldr_bgr, uncertain_mask, reference_bgr, stats)。
    """
    n = len(images)
    h, w = images[0].shape[:2]
    scale = min(1.0, (MAX_PIXELS / (h * w)) ** 0.5)
    if scale < 1.0:
        images = [cv2.resize(im, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA) for im in images]

    aligned, cover, overlap, inliers = _align(images, ref_idx)
    h, w = aligned[0].shape[:2]
    t = [float(x) for x in shutters]

    lum, state, gray8 = [], [], []
    for i in range(n):
        lin = np.power(aligned[i].astype(np.float32) / 255.0, GAMMA)
        lum.append(((lin @ BGR2Y) / t[i]).astype(np.float32))
        del lin
        zmax = aligned[i].max(axis=2)
        s = np.full((h, w), VALID, np.uint8)
        s[zmax >= SAT_HIGH] = SAT_HI
        s[zmax <= DARK_LOW] = SAT_LO
        s[cover[i] == 0] = NOCOV
        state.append(s)
        gray8.append(cv2.cvtColor(aligned[i], cv2.COLOR_BGR2GRAY))

    motion = _detect_motion(lum, state, t)
    hdr, uncertain = _merge(aligned, gray8, state, motion, ref_idx, t)
    ldr = _tonemap(hdr, uncertain)
    mask = np.where(uncertain, 255, 0).astype(np.uint8)

    stats = {
        "frames": n,
        "reference": ref_idx,
        "width": w,
        "height": h,
        "downscaled": bool(scale < 1.0),
        "overlap_pct": round(float(overlap) * 100, 1),
        "motion_pct": round(float(motion.mean()) * 100, 2),
        "uncertain_pct": round(float(uncertain.mean()) * 100, 2),
        "align_inliers": inliers,
    }
    return ldr, mask, aligned[ref_idx], stats


# ---------------------------------------------------------------- 配准

def _align(images, ref_idx):
    """把每张图对齐到参照帧坐标系，返回 (对齐图, 覆盖掩码, 重叠占比, 内点数)。"""
    n = len(images)
    h, w = images[ref_idx].shape[:2]
    s = min(1.0, ALIGN_MAX_SIDE / max(h, w))
    grays = []
    for im in images:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if s < 1.0:
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        grays.append(g)

    orb = cv2.ORB_create(nfeatures=6000)
    kp_r, des_r = orb.detectAndCompute(grays[ref_idx], None)
    if des_r is None or len(kp_r) < 20:
        raise PipelineError("参照帧特征点过少，无法配准")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    full = np.full((h, w), 255, np.uint8)
    aligned = [None] * n
    cover = [None] * n
    inliers = [-1] * n
    aligned[ref_idx] = images[ref_idx]
    cover[ref_idx] = full

    for i in range(n):
        if i == ref_idx:
            continue
        kp, des = orb.detectAndCompute(grays[i], None)
        if des is None or len(kp) < 20:
            raise PipelineError(f"第 {i + 1} 张图片特征点过少，无法与参照帧配准")
        good = []
        for pair in bf.knnMatch(des_r, des, k=2):
            if len(pair) == 2:
                a, b = pair
                if a.distance < 0.75 * b.distance:
                    good.append(a)
        if len(good) < 12:
            raise PipelineError(f"第 {i + 1} 张与参照帧的匹配点不足，无法配准")
        src = np.float32([kp_r[m.queryIdx].pt for m in good])   # 参照帧坐标
        dst = np.float32([kp[m.trainIdx].pt for m in good])     # 第 i 帧坐标
        m, inl = cv2.estimateAffinePartial2D(
            src, dst, cv2.RANSAC, ransacReprojThreshold=3.0)
        if m is None or inl is None or int(inl.sum()) < 10:
            raise PipelineError(f"第 {i + 1} 张与参照帧配准失败")
        if s < 1.0:  # 变换是在降采样坐标系估计的，平移量换算回原分辨率
            m = m.copy()
            m[:, 2] /= s
        aligned[i] = cv2.warpAffine(
            images[i], m, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        cover[i] = cv2.warpAffine(
            full, m, (w, h), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        inliers[i] = int(inl.sum())

    inter = cover[0] > 0
    for c in cover[1:]:
        inter &= c > 0
    frac = float(inter.mean())
    if frac < MIN_OVERLAP:
        raise PipelineError(
            f"配准后公共重叠区域仅 {frac * 100:.0f}%"
            f"（低于 {MIN_OVERLAP * 100:.0f}%），不输出成片")
    return aligned, cover, frac, inliers


# ------------------------------------------------------------ 运动检测

def _detect_motion(lum, state, t):
    """两两一致性检验，返回布尔运动掩码。"""
    n = len(lum)
    h, w = lum[0].shape
    motion = np.zeros((h, w), bool)
    floors = [(3.0 / 255.0) ** GAMMA / ti for ti in t]      # 噪声底（辐照度）
    sat_b = [(SAT_HIGH / 255.0) ** GAMMA / ti for ti in t]  # 高光裁切对应的辐照度下界
    dark_b = [(DARK_LOW / 255.0) ** GAMMA / ti for ti in t]  # 暗部对应的辐照度上界
    # 邻域 min/max：比较时容忍 ±1px 的配准残差
    lmin = [cv2.erode(l, K3) for l in lum]
    lmax = [cv2.dilate(l, K3) for l in lum]

    def sat_core(i, which):
        # 裁切区域向内收缩 1px，避免边界错位造成误判
        return cv2.erode((state[i] == which).astype(np.uint8), K3).astype(bool)

    for i in range(n):
        for j in range(i + 1, n):
            si, sj = state[i], state[j]
            both = (si == VALID) & (sj == VALID)
            if both.any():
                thr = REL_TOL * np.maximum(lum[i], lum[j]) + floors[i] + floors[j]
                motion |= both & ((lum[i] > lmax[j] + thr) | (lum[i] < lmin[j] - thr))
            # 高光裁切只给出下界：另一帧显著更暗 => 场景发生变化
            m = sat_core(i, SAT_HI) & (sj == VALID)
            if m.any():
                motion |= m & (lum[j] < BOUND_MARGIN * sat_b[i])
            m = sat_core(j, SAT_HI) & (si == VALID)
            if m.any():
                motion |= m & (lum[i] < BOUND_MARGIN * sat_b[j])
            # 暗部不可靠只给出上界：另一帧显著更亮 => 场景发生变化
            m = sat_core(i, SAT_LO) & (sj == VALID)
            if m.any():
                motion |= m & (lum[j] > dark_b[i] / BOUND_MARGIN)
            m = sat_core(j, SAT_LO) & (si == VALID)
            if m.any():
                motion |= m & (lum[i] > dark_b[j] / BOUND_MARGIN)

    # 形态学清理：去散点、补小洞、去小连通块，再轻微膨胀覆盖物体边缘
    u = motion.astype(np.uint8) * 255
    u = cv2.morphologyEx(u, cv2.MORPH_OPEN, K3)
    u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, K5)
    nlab, labels, st, _ = cv2.connectedComponentsWithStats(u)
    min_area = max(64, int(0.0002 * h * w))
    keep = np.zeros_like(u)
    for lab in range(1, nlab):
        if st[lab, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == lab] = 255
    return cv2.dilate(keep, K3).astype(bool)


# ---------------------------------------------------------------- 合成

def _merge(aligned, gray8, state, motion, ref_idx, t):
    """静态区域加权平均所有未裁切帧；运动区域只取参照帧。

    返回 (HDR 辐照度图, 不确定掩码)。权重采用经典三角帽函数，
    中间曝光权重高、接近裁切的曝光权重低。
    """
    n = len(aligned)
    h, w = motion.shape
    static = ~motion
    num = np.zeros((h, w, 3), np.float32)
    den = np.zeros((h, w), np.float32)
    for i in range(n):
        valid = state[i] == VALID
        m = valid if i == ref_idx else (valid & static)
        if not m.any():
            continue
        z = gray8[i].astype(np.float32)
        wgt = np.minimum(z, 255.0 - z) / 128.0
        np.clip(wgt, 0.02, 1.0, out=wgt)
        wgt *= m
        lin = np.power(aligned[i].astype(np.float32) / 255.0, GAMMA) / t[i]
        num += lin * wgt[..., None]
        den += wgt
        del lin
    # 没有任何有效曝光的像素：保留参照帧内容并标记为不确定
    ref_lin = np.power(aligned[ref_idx].astype(np.float32) / 255.0, GAMMA) / t[ref_idx]
    uncertain = den < 1e-6
    hdr = np.where(uncertain[..., None], ref_lin,
                   num / np.maximum(den, 1e-9)[..., None])
    return hdr, uncertain


def _tonemap(hdr, uncertain):
    """Reinhard 色调映射（按亮度压缩、保持色彩），输出 8bit BGR。"""
    lum = hdr @ BGR2Y
    certain = ~uncertain
    base = lum[certain] if certain.any() else lum.ravel()
    key = max(float(np.percentile(base, 90)), 1e-4)
    yn = lum / key
    yd = yn / (1.0 + yn)
    out = hdr * (yd / np.maximum(lum, 1e-8))[..., None]
    out = np.clip(out, 0.0, 1.0) ** (1.0 / GAMMA)
    return (out * 255.0 + 0.5).astype(np.uint8)
