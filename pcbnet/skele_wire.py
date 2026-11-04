import cv2
import numpy as np
import heapq
import os
try:  # 当前实现未使用 skeletonize；避免提交镜像缺 scikit-image 时 import 失败
    from skimage.morphology import skeletonize
except Exception:  # noqa: BLE001
    skeletonize = None
from collections import deque
from typing import List, Tuple, Dict, Any, Set, Optional
from tqdm import tqdm
import math
from pathlib import Path
from sklearn.cluster import KMeans
from .llm import fix_pin_by_llm, PIN_NUMBER
import matplotlib.pyplot as plt

# ===================== PCB走线提取配置参数区（统一管理） =====================
TEXT_CM_BORDER_PX = 5
COMP_CM_BORDER_PX = 3
CSP_STEP_CFG = {
    "default": (0, 15),
    "resistor": (0, 10)
}
CSP_EDGE_PAD_PX = 5
CSP_TRY_BOUNDRY = ['ic']
CSP_BACK_SEARCH_COMP = ['capacitor']
EP_DIFF_THRESH = 40
RCP_SCAN_RANGE = 10
RCP_MAX_BACK_STEP = 10

# build_pcb_nets
BN_SCAN_RANGE = 15
BN_THRES = 0.99
BN_DIFF_THRESH = 40
BN_NEAR_WIN = 5
BN_STOP_NEAR_WIN = 5


# 四方向：上、下、左、右
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def _get_wire_binary(
    img: np.ndarray,          # 原始BGR彩色图
    gray: np.ndarray,
    comps: List[Any],
    smooth_win: int = 3,
    offset: int = 5
) -> Tuple[np.ndarray, int]:
    """
    返回：
        binary_raw: 单纯去掉背景的前景二值图 (1=器件+走线前景)
        comp_binary: 器件掩码，bbox内前景像素聚类后保留数量多的一类，少的置0
        wire_binary: binary_raw - comp_binary，最终走线掩码
        final_t: 使用的灰度阈值
    背景高亮白底，走线暗前景
    """
    h, w = gray.shape
    total = h * w
    t_list = np.arange(255, -1, -1)
    r_list = []
    for t in t_list:
        bg_cnt = np.sum(gray > t)
        r_list.append(bg_cnt / total)
    r_arr = np.array(r_list)

    dr = np.diff(r_arr)
    if smooth_win > 1:
        dr = np.convolve(dr, np.ones(smooth_win), "valid") / smooth_win
    ddr = np.diff(dr)
    ddr_min_idx = int(np.argmin(ddr))

    slowdown_t = int(t_list[ddr_min_idx + 1])
    final_t = int(np.clip(slowdown_t - offset, 0, 255))
    binary_raw = np.where(gray <= final_t, 1, 0).astype(np.uint8)

    return binary_raw, final_t


def _get_cross_mask(
    img_shape: Tuple[int, int],
    bboxs: List[Any],
    is_cross: bool,
    cross_width=2,
    border_px=5  # is_cross=False：外围保留border_px边框；is_cross=True：在向内缩border_px后的区域内画米字
) -> np.ndarray:
    """
    mask规则：
    1. 原图mask默认 255
    2. 每个bbox使用原始[x1,y1,x2,y2]
       - is_cross=False：外框保留 border_px 一圈边框，内部全部挖空置0
       - is_cross=True：先向内收缩 border_px 得到内框，**只在这个内框里面画米字**，原bbox其余区域保持255
    """
    h, w = img_shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    half_cross = cross_width // 2

    for ele in bboxs:
        x1, y1, x2, y2 = map(int, ele.bbox)
        # 边界裁剪
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue

        # 向内收缩 border_px 得到内框（米字的有效范围）
        inner_x1 = x1 + border_px
        inner_y1 = y1 + border_px
        inner_x2 = x2 - border_px
        inner_y2 = y2 - border_px
        if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
            continue

        cx = (inner_x1 + inner_x2) / 2.0
        cy = (inner_y1 + inner_y2) / 2.0
        cx_int = int(round(cx))
        cy_int = int(round(cy))

        if not is_cross:
            # 保留外围border_px边框，内部挖空
            mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0
        else:
            # 米字只画在收缩后的内框 [inner_x1, inner_y1, inner_x2, inner_y2] 内部
            # 水平中线
            xc1 = max(0, cx_int - half_cross)
            xc2 = min(w, cx_int + half_cross + 1)
            mask[inner_y1:inner_y2, xc1:xc2] = 0
            # 垂直中线
            yc1 = max(0, cy_int - half_cross)
            yc2 = min(h, cy_int + half_cross + 1)
            mask[yc1:yc2, inner_x1:inner_x2] = 0
            # 两条对角线（内框四角）
            cv2.line(mask, (inner_x1, inner_y1), (inner_x2, inner_y2), 0, thickness=cross_width)
            cv2.line(mask, (inner_x2, inner_y1), (inner_x1, inner_y2), 0, thickness=cross_width)
    return mask


def _split_continuous_segments(vals: List[int]) -> List[List[int]]:
    if not vals:
        return []
    vals = sorted(vals)
    segments = []
    current = [vals[0]]
    for v in vals[1:]:
        if v == current[-1] + 1:
            current.append(v)
        else:
            segments.append(current)
            current = [v]
    segments.append(current)
    return segments


def _process_horizontal_edge(
    wire_binary: np.ndarray,
    x_fixed: int,
    y1: int,
    y2: int,
    start_step: int,
    total_steps: int,
    dx: int,
    side: str,
    guess_boundry_len: float,
) -> List[Tuple[int, int, str]]:
    h, w = wire_binary.shape
    survived_ys = None
    run_cnt = 0
    candidate_points = []
    k = start_step - 1
    while True:
        k += 1
        curr_x = x_fixed + dx * k
        if curr_x < 0 or curr_x >= w:
            break
        y_start = max(0, y1)
        y_end = min(h - 1, y2)
        if y_start >= y_end:
            continue
        col = wire_binary[y_start: y_end + 1, curr_x]
        if np.sum(col > 0) > guess_boundry_len:
            survived_ys = None
            run_cnt = 0
            continue
        curr_ys = np.where(col > 0)[0] + y_start
        curr_ys_set = set(curr_ys.tolist())
        if survived_ys is None:
            survived_ys = curr_ys_set
        else:
            survived_ys = survived_ys.intersection(curr_ys_set)
        run_cnt += 1
        if not survived_ys:
            break
        if run_cnt >= total_steps:
            seq_start_k = k - (total_steps - 1)
            seq_start_x = x_fixed + dx * seq_start_k
            segments = _split_continuous_segments(list(survived_ys))
            for seg in segments:
                avg_y = int(round(np.mean(seg)))
                candidate_points.append((seq_start_x, avg_y, side))
            break
    return candidate_points


def _process_vertical_edge(
    wire_binary: np.ndarray,
    y_fixed: int,
    x1: int,
    x2: int,
    start_step: int,
    total_steps: int,
    dy: int,
    side: str,
    guess_boundry_len: float,
) -> List[Tuple[int, int, str]]:
    h, w = wire_binary.shape
    survived_xs = None
    run_cnt = 0
    candidate_points = []
    k = start_step - 1
    while True:
        k += 1
        curr_y = y_fixed + dy * k
        if curr_y < 0 or curr_y >= h:
            break
        x_start = max(0, x1)
        x_end = min(w - 1, x2)
        if x_start >= x_end:
            continue
        row = wire_binary[curr_y, x_start: x_end + 1]
        if np.sum(row > 0) > guess_boundry_len:
            survived_xs = None
            run_cnt = 0
            continue
        curr_xs = np.where(row > 0)[0] + x_start
        curr_xs_set = set(curr_xs.tolist())
        if survived_xs is None:
            survived_xs = curr_xs_set
        else:
            survived_xs = survived_xs.intersection(curr_xs_set)
        run_cnt += 1
        if not survived_xs:
            break
        if run_cnt >= total_steps:
            seq_start_k = k - (total_steps - 1)
            seq_start_y = y_fixed + dy * seq_start_k
            segments = _split_continuous_segments(list(survived_xs))
            for seg in segments:
                avg_x = int(round(np.mean(seg)))
                candidate_points.append((avg_x, seq_start_y, side))
            break
    return candidate_points


def _find_border_x_y(
    border1,
    border2,
    move,
    try_boundry,
    edge_pad_px,
    wire_binary,
    boundary_ratio_thresh,
    is_vertical: bool,   # True:固定x扫y区间(竖列)；False:固定y扫x区间(横行)
    search_dir: int      # 向外扩张方向 +1 / -1
):
    use_move = None
    h, w = wire_binary.shape[:2]

    # 原有边缘padding逻辑保留
    if border1 + 2 * edge_pad_px < border2:
        border1 = border1 + edge_pad_px
        border2 = border2 - edge_pad_px

    if border1 >= border2 or not try_boundry:
        return border1, border2, move

    # ========== 阶段1：就近探测，按顺序 0, +1, -1, +2, -2, +3, -3, +4, -4, +5, -5 找第一个合格边界点 ==========
    probe_offsets = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
    start_bound = None
    for off in probe_offsets:
        pos = move + off
        # 越界检查
        if is_vertical:
            if not (0 <= pos < w):
                continue
            roi = wire_binary[border1:border2+1, pos]
        else:
            if not (0 <= pos < h):
                continue
            roi = wire_binary[pos, border1:border2+1]

        ratio = np.sum(roi > 0) / roi.size
        if ratio > boundary_ratio_thresh:
            start_bound = pos
            break
    # 就近探测没找到合格点，直接返回原值
    if start_bound is None:
        return border1, border2, move

    # ========== 阶段2：从找到的start_bound，沿着search_dir单向外扩，直到不满足阈值，取上一个合格位置 ==========
    current_pos = start_bound
    last_good = start_bound
    max_extend = 10  # 最大外扩步数，防止死循环，按需调

    for _ in range(max_extend):
        next_pos = current_pos + search_dir
        # 越界保护
        if is_vertical:
            if not (0 <= next_pos < w):
                break
            roi = wire_binary[border1:border2+1, next_pos]
        else:
            if not (0 <= next_pos < h):
                break
            roi = wire_binary[next_pos, border1:border2+1]
        last_good = next_pos
        current_pos = next_pos
        ratio = np.sum(roi > 0) / roi.size
        if ratio < boundary_ratio_thresh:
            break

    use_move = last_good
    if use_move is not None:
        move = use_move
    return border1, border2, move


def _get_comp_start_points(
    wire_binary: np.ndarray,
    comps: List[Any],
    step_cfg: Dict[str, Tuple[int, int]],
    boundary_ratio_thresh: float = 0.9,
    edge_pad_px: int = 5,
    max_edge_pad_ratio: float = 0.1
) -> Dict[int, List[Tuple[int, int, str]]]:
    start_points = {}

    for comp_idx, comp in enumerate(comps):
        start_step, total_steps = step_cfg.get(comp.label, step_cfg['default'])
        x1, y1, x2, y2 = map(int, comp.bbox)
        pts = []
        try_boundry = (comp.label in CSP_TRY_BOUNDRY)

        hor_bound_len = (y2 - y1) * boundary_ratio_thresh
        fix_edge_pad_px = min(int((y2 - y1) * max_edge_pad_ratio), edge_pad_px)
        re_y1, re_y2, use_x = _find_border_x_y(
            border1=y1, border2=y2, move=x2,
            try_boundry=try_boundry, edge_pad_px=fix_edge_pad_px,
            wire_binary=wire_binary, boundary_ratio_thresh=boundary_ratio_thresh,
            is_vertical=True, search_dir=+1
        )
        pts.extend(_process_horizontal_edge(
            wire_binary, use_x, re_y1, re_y2,
            start_step, total_steps, dx=1, side="right",
            guess_boundry_len=hor_bound_len
        ))

        le_y1, le_y2, use_x = _find_border_x_y(
            border1=y1, border2=y2, move=x1,
            try_boundry=try_boundry, edge_pad_px=fix_edge_pad_px,
            wire_binary=wire_binary, boundary_ratio_thresh=boundary_ratio_thresh,
            is_vertical=True, search_dir=-1
        )
        pts.extend(_process_horizontal_edge(
            wire_binary, use_x, le_y1, le_y2,
            start_step, total_steps, dx=-1, side="left",
            guess_boundry_len=hor_bound_len
        ))

        ver_boundry_len = (x2 - x1) * boundary_ratio_thresh
        fix_edge_pad_px = min(int((x2 - x1) * max_edge_pad_ratio), edge_pad_px)
        be_x1, be_x2, use_y = _find_border_x_y(
            border1=x1, border2=x2, move=y2,
            try_boundry=try_boundry, edge_pad_px=fix_edge_pad_px,
            wire_binary=wire_binary, boundary_ratio_thresh=boundary_ratio_thresh,
            is_vertical=False, search_dir=1
        )
        pts.extend(_process_vertical_edge(
            wire_binary, use_y, be_x1, be_x2,
            start_step, total_steps, dy=1, side="bottom",
            guess_boundry_len=ver_boundry_len
        ))

        te_x1, te_x2, use_y = _find_border_x_y(
            border1=x1, border2=x2, move=y1,
            try_boundry=try_boundry, edge_pad_px=fix_edge_pad_px,
            wire_binary=wire_binary, boundary_ratio_thresh=boundary_ratio_thresh,
            is_vertical=False, search_dir=-1
        )
        pts.extend(_process_vertical_edge(
            wire_binary, use_y, te_x1, te_x2,
            start_step, total_steps, dy=-1, side="top",
            guess_boundry_len=ver_boundry_len
        ))

        start_points[comp_idx] = pts
    return start_points


def refine_comp_and_start_ps(
    wire_binary: np.ndarray,
    comps: List[Any],
    comp_start_pts: Dict[int, List[Tuple[int, int, str]]],
    boundary_thres: float = 0.9,
    max_comp_back_ratio: float = 0.5
) -> Tuple[List[Any], Dict[int, List[Tuple[int, int, str]]]]:
    """
    1. 对指定label器件：每个start_point沿着side方向回退扫描，直到 ratio > boundary_thres；
       找到则更新comp.bbox + 该start_ps坐标；一直走到对边都没找到则保持原样
    2. 校验PIN数量：
       - capacitor / resistor：固定2个pin，必须在不同边框；同坐标轴相近；缺则以现有点镜像到对边补点
       - ground：固定1个pin；必须落在该边中心30%区间内，不在则剔除/修正
       - pin / net_short：如果点位数量等于3，直接剔除该器件以及对应的点位
    """
    new_comps = []
    # 拷贝原始点位字典
    new_start_pts = {k: [p for p in v] for k, v in comp_start_pts.items()}
    side_back_dir = {
        "left": (1, 0),
        "right": (-1, 0),
        "top": (0, 1),
        "bottom": (0, -1)
    }
    side_opposite = {
        "left": ("right", "y"),
        "right": ("left", "y"),
        "top": ("bottom", "x"),
        "bottom": ("top", "x"),
    }

    for comp_idx, comp in enumerate(comps):
        label = comp.label
        pts = new_start_pts.get(comp_idx, [])

        if label in ("pin", "net_short") and len(pts) == 3:
            if comp_idx in new_start_pts:
                del new_start_pts[comp_idx]
            # 不加入 new_comps，并且后面所有逻辑跳过
            continue

        target_pin_cnt = PIN_NUMBER.get(label, None)
        if not target_pin_cnt or not pts:
            new_comps.append(comp)
            continue

        x1, y1, x2, y2 = map(int, comp.bbox)
        h_box = y2 - y1
        w_box = x2 - x1

        if len(pts) != target_pin_cnt:
            if label in ("capacitor", "resistor"):
                final_pts = pts[:]
                # 大于2个，边的对边映射 + 固定比较轴
                if len(final_pts) > 2:
                    keep = []
                    for idx, p in enumerate(final_pts):
                        px, py, s = p
                        opp_side, axis = side_opposite[s]
                        has_mate = False
                        for jdx, q in enumerate(final_pts):
                            if idx == jdx:
                                continue
                            qx, qy, qs = q
                            if qs != opp_side:
                                continue
                            if axis == "y":
                                diff = abs(py - qy)
                            else:
                                diff = abs(px - qx)
                            if diff < 3:
                                has_mate = True
                                break
                        if has_mate:
                            keep.append(p)
                    if len(keep) > 0:
                        final_pts = keep

                # 不足2个：镜像补点到对边
                elif len(final_pts) == 1:
                    x0, y0, s0 = final_pts[0]
                    if s0 == "left":
                        new_p = (x2, y0, "right")
                    elif s0 == "right":
                        new_p = (x1, y0, "left")
                    elif s0 == "top":
                        new_p = (x0, y2, "bottom")
                    elif s0 == "bottom":
                        new_p = (x0, y1, "top")
                    else:
                        new_p = None
                    if new_p is not None:
                        final_pts.append(new_p)

                new_start_pts[comp_idx] = final_pts

            elif label == "ground":
                # 规则：仅1个pin；落在该边中心30%区间
                final_pts = []
                for (px, py, side) in pts:
                    ok = False
                    if side in ("left", "right"):
                        h_seg = y2 - y1
                        mid_y = (y1 + y2) / 2
                        low = mid_y - 0.15 * h_seg
                        high = mid_y + 0.15 * h_seg
                        if low <= py <= high:
                            ok = True
                    else:
                        w_seg = x2 - x1
                        mid_x = (x1 + x2) / 2
                        low = mid_x - 0.15 * w_seg
                        high = mid_x + 0.15 * w_seg
                        if low <= px <= high:
                            ok = True
                    if ok:
                        final_pts.append((px, py, side))
                new_start_pts[comp_idx] = final_pts

        new_comps.append(comp)

        if label in CSP_BACK_SEARCH_COMP:
            updated_pts = []

            for (px, py, side) in pts:
                dx_back, dy_back = side_back_dir[side]
                cx, cy = px, py
                hit = False
                while True:
                    if side == "left" and cx >= x1 + max_comp_back_ratio * w_box:
                        break
                    if side == "right" and cx <= x2 - max_comp_back_ratio * w_box:
                        break
                    if side == "top" and cy >= y1 + max_comp_back_ratio * h_box:
                        break
                    if side == "bottom" and cy <= y2 - max_comp_back_ratio * h_box:
                        break

                    if side in ("left", "right"):
                        col = wire_binary[y1:y2+1, cx]
                        ratio = np.sum(col > 0) / len(col)
                        if ratio > boundary_thres:
                            hit = True
                            break
                        cx += dx_back
                    else:
                        row = wire_binary[cy, x1:x2+1]
                        ratio = np.sum(row > 0) / len(row)
                        if ratio > boundary_thres:
                            hit = True
                            break
                        cy += dy_back
                if hit:
                    new_pt = (cx, cy, side)
                    updated_pts.append(new_pt)
                    if side == "left":
                        comp.bbox = (cx, y1, x2, y2)
                        x1 = cx
                    elif side == "right":
                        comp.bbox = (x1, y1, cx, y2)
                        x2 = cx
                    elif side == "top":
                        comp.bbox = (x1, cy, x2, y2)
                        y1 = cy
                    elif side == "bottom":
                        comp.bbox = (x1, y1, x2, cy)
                        y2 = cy
                else:
                    updated_pts.append((px, py, side))
            new_start_pts[comp_idx] = updated_pts
            pts = updated_pts

    return new_comps, new_start_pts


# def extract_comp_pins_straight_hsv(
#     bgr_img: np.ndarray,
#     comp_start_pts: Dict[int, List[Tuple[int, int, str]]],
#     wire_binary: np.ndarray,
#     diff_thresh: int = 20,
#     max_step: int = 2000
# ) -> Tuple[Dict[int, List[Dict[str, Any]]], np.ndarray]:
#     """仅沿初始方向直线追踪，不拐弯，HSV色相环形差值判定
#     修改：以 path[-3] 为中心，前后各2px（path[-5] / path[-1]）；
#     增加前置条件：path[-5]、path[-4] 的 wire_binary 必须为1
#     优先级：meet相遇 > gradient色相跳变 > dead_end断线
#     """
#     h, w = bgr_img.shape[:2]
#     lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
#     L = lab[...,0]
#     A = lab[...,1]
#     B = lab[...,2]
#     side2dir = {"right": (1, 0), "left": (-1, 0), "bottom": (0, 1), "top": (0, -1)}

#     # map: (起点x,y) -> (器件cid, side)
#     start_point_map: Dict[Tuple[int, int], Tuple[int, str]] = {}
#     for cid, pts in comp_start_pts.items():
#         for x, y, side in pts:
#             start_point_map[(x, y)] = (cid, side)

#     comp_pins: Dict[int, List[Dict[str, Any]]] = {cid: [] for cid in comp_start_pts.keys()}
#     handled_pairs = set()
#     delta_map = np.full((h, w), np.nan, dtype=np.float32)

#     for comp_idx, start_pts in comp_start_pts.items():
#         for (sx, sy, side) in start_pts:
#             dx, dy = side2dir[side]
#             cx, cy = sx, sy
#             path: List[Tuple[int, int]] = []
#             step = 0

#             while 0 <= cx < w and 0 <= cy < h and step < max_step:
#                 if wire_binary[cy, cx] <= 0:
#                     break
#                 path.append((cx, cy))
#                 step += 1

#                 # ========== 【优先级1：相遇meet检测，放到最前面】 ==========
#                 if (cx, cy) in start_point_map:
#                     other_cid, other_side = start_point_map[(cx, cy)]
#                     if other_cid != comp_idx:
#                         # 无序配对key，防止双向重复处理
#                         key = frozenset({(sx, sy, comp_idx, side), (cx, cy, other_cid, other_side)})
#                         if key not in handled_pairs:
#                             handled_pairs.add(key)
#                             mid_idx = len(path) // 2
#                             mx, my = path[mid_idx]
#                             comp_pins[comp_idx].append({"x": mx, "y": my, "type": "meet", "side": side})
#                             comp_pins[other_cid].append({"x": mx, "y": my, "type": "meet", "side": other_side})
#                         break

#                 # ========== 【优先级2：gradient色相跳变检测】 ==========
#                 # 中心：path[-3]，前2=path[-5]，后2=path[-1]，至少需要5个点
#                 if len(path) >= 5:
#                     # 取5个位置
#                     x5, y5 = path[-5]
#                     x4, y4 = path[-4]
#                     x3, y3 = path[-3]   # 中心点（输出坐标）
#                     x2, y2 = path[-2]
#                     x1, y1 = path[-1]

#                     # 前置校验：左右两组都必须在铜线上
#                     if (wire_binary[y5, x5] > 0 and wire_binary[y4, x4] > 0 and
#                         wire_binary[y2, x2] > 0 and wire_binary[y1, x1] > 0):

#                         # ========== 沿走线方向 1×2 线型窗口采样：(x,y) 和向前一步(x+dx,y+dy) ==========
#                         def get_lab_line_win(x, y, dx_, dy_):
#                             pts = []
#                             for off in [(0,0), (dx_, dy_)]:
#                                 nx_p = x + off[0]
#                                 ny_p = y + off[1]
#                                 nx_p = np.clip(nx_p, 0, w-1)
#                                 ny_p = np.clip(ny_p, 0, h-1)
#                                 pts.append((nx_p, ny_p))
#                             Ls = [L[yy, xx] for xx, yy in pts]
#                             As = [A[yy, xx] for xx, yy in pts]
#                             Bs = [B[yy, xx] for xx, yy in pts]
#                             return np.median(Ls), np.median(As), np.median(Bs)

#                         L5, A5, B5 = get_lab_line_win(x5, y5, dx, dy)
#                         L4, A4, B4 = get_lab_line_win(x4, y4, dx, dy)
#                         # 左组整体中值
#                         L_left = np.median([L5, L4])
#                         A_left = np.median([A5, A4])
#                         B_left = np.median([B5, B4])

#                         # ========== 右组：path[-2], path[-1] ==========
#                         L2, A2, B2 = get_lab_line_win(x2, y2, dx, dy)
#                         L1, A1, B1 = get_lab_line_win(x1, y1, dx, dy)
#                         L_right = np.median([L2, L1])
#                         A_right = np.median([A2, A1])
#                         B_right = np.median([B2, B1])

#                         # ΔE76
#                         dL = L_left - L_right
#                         dA = A_left - A_right
#                         dB = B_left - B_right
#                         delta = np.sqrt(dL*dL + dA*dA + dB*dB)
#                         # print(f"center ({x3},{y3}), delta={delta:.2f}")
#                         delta_map[y3, x3] = delta

#                         if delta > diff_thresh:
#                             # 中心点 path[-3] 作为梯度点
#                             comp_pins[comp_idx].append({"x": x3, "y": y3, "type": "gradient", "side": side})
#                             break


#                 # ========== 【优先级3：判断下一步是否断线 dead_end】 ==========
#                 nx = cx + dx
#                 ny = cy + dy
#                 if not (0 <= nx < w and 0 <= ny < h and wire_binary[ny, nx] > 0):
#                     comp_pins[comp_idx].append({"x": cx, "y": cy, "type": "dead_end", "side": side})
#                     break
#                 cx, cy = nx, ny
#             # end while
#     return comp_pins, delta_map


def plot_delta_heatmap(bgr_img: np.ndarray, delta_map: np.ndarray, diff_thresh: float, save_path=None):
    """
    叠加原图 + delta热力图
    :param bgr_img: 原始BGR图
    :param delta_map: extract 返回的delta_map，nan=无值
    :param diff_thresh: 阈值，在色标上标注
    :param save_path: 保存图片路径，None则直接show
    """
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    delta_abs = np.abs(delta_map)  # 取绝对值

    plt.figure(figsize=(12, 9))
    plt.imshow(rgb, alpha=0.25)  # 背景透明度调低（原图更淡）
    im = plt.imshow(delta_abs, cmap="jet", vmin=0, vmax=diff_thresh * 0.6, alpha=1)
    cbar = plt.colorbar(im)
    cbar.set_label("|ΔE76|", fontsize=11)
    cbar.ax.axhline(y=diff_thresh, color="white", linestyle="--", label=f"thresh={diff_thresh}")
    cbar.ax.legend()
    plt.title("Delta Heatmap |Gradient Center Points|", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")


def _cluster_1d(values: List[int], eps: int = 10) -> List[List[int]]:
    """
    对一维坐标做聚类，返回每个簇中元素的下标列表。
    排序后扫描，相邻值差距 <= eps 归为同一簇（基于间隙分裂）。
    """
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: values[i])
    clusters: List[List[int]] = []
    cur = [order[0]]
    for k in range(1, n):
        i = order[k]
        prev = order[k - 1]
        if values[i] - values[prev] <= eps:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)
    return clusters


def refine_and_cluster_pins(
    comp_pins: Dict[int, List[Dict[str, Any]]],
    comp_start_pts: Dict[int, List[Tuple[int, int, str]]],
    wire_binary: np.ndarray,
    scan_range: int = 15,
    max_back_step: int = 120,
    thres: float = 0.93
) -> Dict[int, List[Dict[str, Any]]]:
    """
    独立模块：
    1. 沿追踪方向【双向】（后退+前进）扫描找分叉交点，吸附到交点簇中点
       - type=meet 的pin：跳过回退吸附
       - 回退方向不能越过自身对应的 start_point；前进方向无start约束
    2. 收集全部pin（包含meet），一维x/y聚类，簇内坐标取平均
    """
    h, w = wire_binary.shape
    side2dir = {"right": (1, 0), "left": (-1, 0), "bottom": (0, 1), "top": (0, -1)}

    # 构建 (cid, side) -> 该方向所有起点坐标集合
    cid_side_to_starts: Dict[Tuple[int, str], Set[Tuple[int, int]]] = {}
    for cid, pts in comp_start_pts.items():
        for (sx, sy, side) in pts:
            key = (cid, side)
            if key not in cid_side_to_starts:
                cid_side_to_starts[key] = set()
            cid_side_to_starts[key].add((sx, sy))

    # Step1：逐个pin做双向扫描找交点候选
    for cid, pin_list in comp_pins.items():
        for pin in pin_list:
            ptype = pin.get("type")
            side = pin.get("side")
            # meet 直接跳过吸附
            if ptype == "meet":
                continue
            if side not in side2dir:
                continue

            dx_track, dy_track = side2dir[side]
            rd = (-dx_track, -dy_track)   # 后退方向（朝向器件）
            fd = (dx_track, dy_track)     # 前进方向（远离器件）
            orig_cx, orig_cy = pin["x"], pin["y"]
            fork_candidates = []
            start_set = cid_side_to_starts.get((cid, side), set())

            # ---------- 1）后退方向扫描 ----------
            cx, cy = orig_cx, orig_cy
            step = 0
            while step < max_back_step:
                if not (0 <= cx < w and 0 <= cy < h):
                    break
                if wire_binary[cy, cx] == 0:
                    break
                # 后退方向：碰到器件起点就停，不能越过
                if (cx + 3 * rd[0], cy + 3 * rd[1]) in start_set:
                    break

                is_intersect = False
                if dx_track != 0:
                    # 水平追踪方向 → 正交竖直扫描
                    fixed_x = cx
                    y_low = max(0, cy - scan_range)
                    y_high = min(h - 1, cy + scan_range)
                    if wire_binary[y_low:cy+1, fixed_x].sum() > scan_range * thres or \
                       wire_binary[cy:y_high+1, fixed_x].sum() > scan_range * thres:
                        is_intersect = True
                else:
                    # 竖直追踪方向 → 正交水平扫描
                    fixed_y = cy
                    x_low = max(0, cx - scan_range)
                    x_high = min(w - 1, cx + scan_range)
                    if wire_binary[fixed_y, x_low:cx+1].sum() > scan_range * thres or \
                       wire_binary[fixed_y, cx:x_high+1].sum() > scan_range * thres:
                        is_intersect = True
                if is_intersect:
                    fork_candidates.append((cx, cy))

                cx += rd[0]
                cy += rd[1]
                step += 1

            # ---------- 2）前进方向扫描 ----------
            cx, cy = orig_cx, orig_cy
            step = 0
            while step < max_back_step:
                if not (0 <= cx < w and 0 <= cy < h):
                    break
                if wire_binary[cy, cx] == 0:
                    break
                # 前进方向：不判断start_set，直接顺着走线往前走

                is_intersect = False
                if dx_track != 0:
                    fixed_x = cx
                    y_low = max(0, cy - scan_range)
                    y_high = min(h - 1, cy + scan_range)
                    if wire_binary[y_low:cy+1, fixed_x].sum() > scan_range * thres or \
                       wire_binary[cy:y_high+1, fixed_x].sum() > scan_range * thres:
                        is_intersect = True
                else:
                    fixed_y = cy
                    x_low = max(0, cx - scan_range)
                    x_high = min(w - 1, cx + scan_range)
                    if wire_binary[fixed_y, x_low:cx+1].sum() > scan_range * thres or \
                       wire_binary[fixed_y, cx:x_high+1].sum() > scan_range * thres:
                        is_intersect = True
                if is_intersect:
                    fork_candidates.append((cx, cy))

                cx += fd[0]
                cy += fd[1]
                step += 1

            # 吸附更新坐标
            if fork_candidates:
                xs = [p[0] for p in fork_candidates]
                ys = [p[1] for p in fork_candidates]
                pin["x"] = int(round(np.mean(xs)))
                pin["y"] = int(round(np.mean(ys)))

    # Step2：收集全部pin（meet也参与聚类）一维聚类修正
    # all_pins_flat = []
    # pin_meta = []
    # for cid, pin_list in comp_pins.items():
    #     for idx, pin in enumerate(pin_list):
    #         all_pins_flat.append((pin["x"], pin["y"]))
    #         pin_meta.append((cid, idx))
    # if not all_pins_flat:
    #     return comp_pins

    # xs = [p[0] for p in all_pins_flat]
    # ys = [p[1] for p in all_pins_flat]
    # x_clusters = _cluster_1d(xs, eps=cluster_eps)
    # y_clusters = _cluster_1d(ys, eps=cluster_eps)

    # for cl in x_clusters:
    #     coords = [all_pins_flat[i] for i in cl]
    #     mean_x = int(round(np.mean([c[0] for c in coords])))
    #     for i in cl:
    #         cid, pidx = pin_meta[i]
    #         comp_pins[cid][pidx]["x"] = mean_x
    # for cl in y_clusters:
    #     coords = [all_pins_flat[i] for i in cl]
    #     mean_y = int(round(np.mean([c[1] for c in coords])))
    #     for i in cl:
    #         cid, pidx = pin_meta[i]
    #         comp_pins[cid][pidx]["y"] = mean_y
    return comp_pins


def trace_and_collect_path(
    sx: int,
    sy: int,
    dx: int,
    dy: int,
    wire_binary: np.ndarray,
    L: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    w: int,
    h: int,
    max_step: int,
    scan_range: int,
    branch_sum_thres: float,
    delta_map: np.ndarray,
    all_start_points: List[Tuple[int, int, int, str]],
    comp_idx: int,
    near_start_thresh: int
) -> Tuple[List[Tuple[int, int]], List[float], List[int], str, Optional[Tuple[int, int, int, str]]]:
    """
    单条直线追踪
    返回：path, path_deltas, branch_cnt_list, stop_reason, hit_other_start
    stop_reason: 'wire_zero' / 'near_other_start' / 'max_step'
    hit_other_start: 只有 stop_reason=near_other_start 时不为None，是路径上碰到的那个start_pt
    """
    path: List[Tuple[int, int]] = []
    path_deltas: List[float] = []
    branch_cnt_list: List[int] = []
    cx, cy = sx, sy
    step = 0
    stop_reason = "max_step"
    hit_other_start: Optional[Tuple[int, int, int, str]] = None

    def get_lab_line_win(x, y, dx_, dy_):
        pts = []
        for off in [(0, 0), (dx_, dy_)]:
            nx_p = x + off[0]
            ny_p = y + off[1]
            nx_p = np.clip(nx_p, 0, w - 1)
            ny_p = np.clip(ny_p, 0, h - 1)
            pts.append((nx_p, ny_p))
        Ls = [L[yy, xx] for xx, yy in pts]
        As = [A[yy, xx] for xx, yy in pts]
        Bs = [B[yy, xx] for xx, yy in pts]
        return np.median(Ls), np.median(As), np.median(Bs)

    def scan_perpendicular_continuous(x0, y0, dx_, dy_):
        if dx_ != 0:
            up_len = 0
            yy = y0 - 1
            while yy >= 0 and abs(yy - y0) <= scan_range and wire_binary[yy, x0] == 1:
                up_len += 1
                yy -= 1
            down_len = 0
            yy = y0 + 1
            while yy < h and abs(yy - y0) <= scan_range and wire_binary[yy, x0] == 1:
                down_len += 1
                yy += 1
            is_intersect = (up_len > scan_range * branch_sum_thres) or (down_len > scan_range * branch_sum_thres)
        else:
            left_len = 0
            xx = x0 - 1
            while xx >= 0 and abs(xx - x0) <= scan_range and wire_binary[y0, xx] == 1:
                left_len += 1
                xx -= 1
            right_len = 0
            xx = x0 + 1
            while xx < w and abs(xx - x0) <= scan_range and wire_binary[y0, xx] == 1:
                right_len += 1
                xx += 1
            is_intersect = (left_len > scan_range * branch_sum_thres) or (right_len > scan_range * branch_sum_thres)
        bc = 1 if is_intersect else 0
        return bc

    while 0 <= cx < w and 0 <= cy < h and step < max_step:
        if wire_binary[cy, cx] <= 0:
            stop_reason = "wire_zero"
            break

        # 检测当前路径点是否靠近别的start point
        hit: Optional[Tuple[int, int, int, str]] = None
        for (ox, oy, ocid, oside) in all_start_points:
            if ocid == comp_idx and (ox, oy) == (sx, sy):
                continue
            dist = np.hypot(cx - ox, cy - oy)
            if dist < near_start_thresh:
                hit = (ox, oy, ocid, oside)
                break
        if hit is not None:
            stop_reason = "near_other_start"
            hit_other_start = hit
            break

        path.append((cx, cy))
        current_delta = np.nan

        if len(path) >= 5:
            x5, y5 = path[-5]
            x4, y4 = path[-4]
            x3, y3 = path[-3]
            x2, y2 = path[-2]
            x1, y1 = path[-1]
            if (wire_binary[y5, x5] > 0 and wire_binary[y4, x4] > 0 and
                wire_binary[y2, x2] > 0 and wire_binary[y1, x1] > 0):
                L5, A5, B5 = get_lab_line_win(x5, y5, dx, dy)
                L4, A4, B4 = get_lab_line_win(x4, y4, dx, dy)
                L_left = np.median([L5, L4])
                A_left = np.median([A5, A4])
                B_left = np.median([B5, B4])

                L2, A2, B2 = get_lab_line_win(x2, y2, dx, dy)
                L1, A1, B1 = get_lab_line_win(x1, y1, dx, dy)
                L_right = np.median([L2, L1])
                A_right = np.median([A2, A1])
                B_right = np.median([B2, B1])

                dL = L_left - L_right
                dA = A_left - A_right
                dB = B_left - B_right
                delta = np.sqrt(dL*dL + dA*dA + dB*dB)
                current_delta = delta
                delta_map[y3, x3] = delta

        path_deltas.append(current_delta)
        bc = scan_perpendicular_continuous(cx, cy, dx, dy)
        branch_cnt_list.append(bc)

        nx = cx + dx
        ny = cy + dy
        if not (0 <= nx < w and 0 <= ny < h and wire_binary[ny, nx] > 0):
            stop_reason = "wire_zero"
            break
        cx, cy = nx, ny
        step += 1

    return path, path_deltas, branch_cnt_list, stop_reason, hit_other_start


def classify_pin_from_path(
    path: List[Tuple[int, int]],
    path_deltas: List[float],
    branch_cnt_list: List[int],
    stop_reason: str,
    hit_other_start: Optional[Tuple[int, int, int, str]],
    sx: int,
    sy: int,
    side: str,
    comp_idx: int,
    handled_pairs: set,
    delta_dev_thresh: float,
    grad_end_near_judge: int = 5,
    max_non_bent_steps: int = 100
) -> List[Dict[str, Any]]:
    pins: List[Dict[str, Any]] = []
    if len(path) == 0:
        return pins

    bc_arr = np.array(branch_cnt_list)
    delta_arr = np.array(path_deltas)

    valid_delta = delta_arr[~np.isnan(delta_arr)]
    med_delta = np.median(valid_delta) if len(valid_delta) > 0 else 0.0

    # 找第一个颜色跳变点
    jump_idx = None
    # print(delta_arr - med_delta)
    # print(delta_arr)
    # import pdb; pdb.set_trace()
    for i, d in enumerate(delta_arr):
        if np.isnan(d):
            continue
        if abs(d - med_delta) > delta_dev_thresh:
            jump_idx = i
            break

    # -------- 优先颜色分界跳变流程 --------
    if jump_idx is not None:
        jx, jy = path[jump_idx]
        end_x, end_y = path[-1]
        dist_jump_to_end = np.hypot(jx - end_x, jy - end_y)

        trigger_grad = False
        if stop_reason == "near_other_start":
            trigger_grad = True
        elif stop_reason == "wire_zero" and dist_jump_to_end > grad_end_near_judge:
            trigger_grad = True

        if trigger_grad:
            pins.append({"x": jx, "y": jy, "type": "gradient", "side": side})
            return pins

    # -------- 颜色跳变分支未触发，进入下面 --------
    
    if stop_reason == "near_other_start" and hit_other_start is not None:
        # 配对：当前起点 <-> 追踪路径上命中的那个start
        ox, oy, ocid, oside = hit_other_start
        mx = (sx + ox) // 2
        my = (sy + oy) // 2
        if np.all(bc_arr == 0):
            key = frozenset({(sx, sy, comp_idx, side), (ox, oy, ocid, oside)})
            if key not in handled_pairs:
                handled_pairs.add(key)
                pins.append({"x": mx, "y": my, "type": "meet", "side": side})
                pins.append({"x": mx, "y": my, "type": "meet", "side": oside})
        else:
            pins.append({"x": mx, "y": my, "type": "grad", "side": side})
        return pins

    if stop_reason == "wire_zero":
        if len(bc_arr) < max_non_bent_steps:
            if len(bc_arr) > grad_end_near_judge:
                abc_arr = bc_arr[-grad_end_near_judge:]
            else:
                abc_arr = bc_arr
            no_branch = np.all(abc_arr == 0)
            ex, ey = path[-1]
            if no_branch:
                # wire_zero / max_step，无分叉 → dead_end
                pins.append({"x": ex, "y": ey, "type": "dead_end", "side": side})
            else:
                pins.append({"x": ex, "y": ey, "type": "grad", "side": side})
    
        else:
            # if len(delta_arr) > grad_end_near_judge:
            #     delta_arr = delta_arr[:-grad_end_near_judge]
            # # 存在分叉分支：取全局最大delta作为gradient
            # valid_mask = ~np.isnan(delta_arr)
            # valid_idx = np.nonzero(valid_mask)[0]
            # if len(valid_idx) == 0:
            #     gx, gy = path[-1]
            # else:
            #     sub_idx = np.argmax(delta_arr[valid_mask])
            #     grad_idx = valid_idx[sub_idx]
            #     gx, gy = path[grad_idx]
            gx, gy = path[0]
            pins.append({"x": gx, "y": gy, "type": "gradient", "side": side})

    return pins


def extract_comp_pins_straight_hsv(
    bgr_img: np.ndarray,
    comp_start_pts: Dict[int, List[Tuple[int, int, str]]],
    wire_binary: np.ndarray,
    max_step: int = 2000,
    scan_range: int = 15,
    branch_sum_thres: float = 0.99,
    delta_dev_thresh: float = 8.0,
    near_start_thresh: int = 5,
    grad_end_near_judge: int = 5
) -> Tuple[Dict[int, List[Dict[str, Any]]], np.ndarray]:
    h, w = bgr_img.shape[:2]
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    A = lab[..., 1]
    B = lab[..., 2]
    side2dir = {"right": (1, 0), "left": (-1, 0), "bottom": (0, 1), "top": (0, -1)}

    all_start_points: List[Tuple[int, int, int, str]] = []
    for cid, pts in comp_start_pts.items():
        for x, y, side in pts:
            all_start_points.append((x, y, cid, side))

    comp_pins: Dict[int, List[Dict[str, Any]]] = {cid: [] for cid in comp_start_pts.keys()}
    handled_pairs = set()
    delta_map = np.full((h, w), np.nan, dtype=np.float32)

    for comp_idx, start_pts in comp_start_pts.items():
        for (sx, sy, side) in start_pts:
            dx, dy = side2dir[side]
            path, path_deltas, bc_list, stop_reason, hit_other_start = trace_and_collect_path(
                sx=sx, sy=sy, dx=dx, dy=dy, wire_binary=wire_binary,
                L=L, A=A, B=B, w=w, h=h, max_step=max_step,
                scan_range=scan_range, branch_sum_thres=branch_sum_thres,
                delta_map=delta_map,
                all_start_points=all_start_points,
                comp_idx=comp_idx,
                near_start_thresh=near_start_thresh
            )
            pin_list = classify_pin_from_path(
                path=path,
                path_deltas=path_deltas,
                branch_cnt_list=bc_list,
                stop_reason=stop_reason,
                hit_other_start=hit_other_start,
                sx=sx, sy=sy, side=side, comp_idx=comp_idx,
                handled_pairs=handled_pairs,
                delta_dev_thresh=delta_dev_thresh,
                grad_end_near_judge=grad_end_near_judge,
                max_non_bent_steps= h * 0.2
            )
            for p in pin_list:
                if p["side"] == side:
                    comp_pins[comp_idx].append(p)
    return comp_pins, delta_map


def prepare_pin_data(
    comp_pins: Dict[int, List[Dict[str, Any]]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[Tuple[int, int], List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    """
    引脚预处理：打全局gid
    dead_end 归入 normal_pins；meet 单独分出
    不再返回 dead_pins
    """
    all_pins = []
    gid = 0
    for cid, pin_list in comp_pins.items():
        for pin_idx, pin in enumerate(pin_list):
            full_id = f"comp_{cid}.pin_{pin_idx}"
            info = {
                "cid": cid,
                "pin_idx": pin_idx,
                "x": int(pin["x"]),
                "y": int(pin["y"]),
                # 起点（器件 bbox 边界处的真实引脚位置）；x/y 是沿导线追踪/吸附后的节点位置
                "start_x": int(pin.get("start_x", pin["x"])),
                "start_y": int(pin.get("start_y", pin["y"])),
                "type": pin.get("type", ""),
                "side": pin.get("side", ""),
                "full_id": full_id,
                "gid": gid
            }
            all_pins.append(info)
            gid += 1

    # meet单独，其余（含dead_end）全部归normal
    meet_pins = [p for p in all_pins if p["type"] == "meet"]
    normal_pins = [p for p in all_pins if p["type"] != "meet"]

    pos_to_pins: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for p in normal_pins:
        pos = (p["x"], p["y"])
        pos_to_pins.setdefault(pos, []).append(p)

    pins_dict: Dict[str, Dict[str, Any]] = {}
    for cid, pin_list in comp_pins.items():
        comp_key = f"comp_{cid}"
        pins_dict[comp_key] = {}
        for idx, pin in enumerate(pin_list):
            pins_dict[comp_key][f"pin_{idx}"] = {
                # x/y 是沿引脚引线追踪后的位置（HSV 颜色跳变处 ≈ 引脚端点，与 GT 一致）；
                # start_x/start_y 仅保留 bbox 边界起点信息
                "point": {"x": int(pin["x"]), "y": int(pin["y"])}
            }
    return meet_pins, normal_pins, pos_to_pins, pins_dict


def build_meet_pin_nets(meet_pins: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    2. meet引脚单独组网，无edge
    """
    nets = []
    meet_pos_map: Dict[Tuple[int, int], List[Dict]] = {}
    for p in meet_pins:
        pos = (p["x"], p["y"])
        meet_pos_map.setdefault(pos, []).append(p)
    for pos, group in meet_pos_map.items():
        pin_ids = [p["full_id"] for p in group]
        nets.append({
            "hyperGraph": f"({','.join(pin_ids)})",
            "edges": {}
        })
    return nets


def build_obstacle_mask(
    h: int, w: int,
    comps: List[Any],
    comp_start_pts: Dict[int, List[Tuple[int, int, str]]]
) -> Tuple[np.ndarray, Set[Tuple[int, int]]]:
    """
    3. 构造器件bbox掩码 + start_pt集合
    """
    comp_mask = np.zeros((h, w), dtype=np.uint8)
    for comp in comps:
        x1, y1, x2, y2 = map(int, comp.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        comp_mask[y1:y2+1, x1:x2+1] = 255

    start_pt_set: Set[Tuple[int, int]] = set()
    for cid, pts in comp_start_pts.items():
        for (x, y, _) in pts:
            start_pt_set.add((x, y))
    return comp_mask, start_pt_set


def orth_fork_check(
    x: int, y: int, dx: int, dy: int,
    wire_binary: np.ndarray,
    bgr_img: np.ndarray,
    scan_range: int,
    thres: float,
    diff_thresh: int
) -> tuple[bool, int]:
    h, w = wire_binary.shape

    def get_lab_pixel(x: int, y: int) -> tuple[float, float, float]:
        """单点BGR转LAB，适合少量采样（fork检测这种少量点很合适）
        bgr_img: BGR通道顺序 (H,W,3)
        """
        h, w = bgr_img.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return np.nan, np.nan, np.nan
        # 单点转LAB
        bgr_pix = bgr_img[y, x, :].reshape(1, 1, 3).astype(np.float32) / 255.0
        lab_pix = cv2.cvtColor(bgr_pix, cv2.COLOR_BGR2LAB)
        L, A, B = lab_pix[0,0]
        return float(L), float(A), float(B)

    # 前进、左、右 正交方向
    fd_dir = (dx, dy)
    left_dir = (-dy, dx)
    right_dir = (dy, -dx)
    dirs = [fd_dir, left_dir, right_dir]
    ok = []
    end_pos = []  # 每个方向 scan_range 远端坐标

    for (ddx, ddy) in dirs:
        sum_val = 0
        ex, ey = x, y
        for s in range(1, scan_range + 1):
            nx = x + ddx * s
            ny = y + ddy * s
            if 0 <= nx < w and 0 <= ny < h:
                sum_val += wire_binary[ny, nx]
                ex, ey = nx, ny
            else:
                break
        ok.append(sum_val > scan_range * thres)
        end_pos.append((ex, ey))

    thres_cnt = sum(ok)
    fd_ok, left_ok, right_ok = ok
    (fx, fy), (lx, ly), (rx, ry) = end_pos

    # ========= 规则分支 =========
    if fd_ok and left_ok and right_ok:
        # 直接取最远像素点的 LAB，不带 window
        L_curr, A_curr, B_curr = get_lab_pixel(x, y)
        L_fd,   A_fd,   B_fd   = get_lab_pixel(fx, fy)
        L_left, A_left, B_left = get_lab_pixel(lx, ly)
        L_right,A_right,B_right= get_lab_pixel(rx, ry)

        def calc_delta(l1, a1, b1, l2, a2, b2):
            dL = l1 - l2
            dA = a1 - a2
            dB = b1 - b2
            return np.sqrt(dL * dL + dA * dA + dB * dB)

        delta_fd   = calc_delta(L_curr, A_curr, B_curr, L_fd,   A_fd,   B_fd)
        delta_left = calc_delta(L_curr, A_curr, B_curr, L_left, A_left, B_left)
        delta_right= calc_delta(L_curr, A_curr, B_curr, L_right,A_right,B_right)

        # 前、左、右任意一个方向与当前点颜色差异不足 → return False
        if (delta_fd < diff_thresh
                or delta_left < diff_thresh
                or delta_right < diff_thresh):
            return False, thres_cnt

        return True, thres_cnt

    else:
        # 三者不同时满足：左右任意 ok 就 True，都不 ok False
        return left_ok or right_ok, thres_cnt


def cross_shape_check(
    x: int, y: int,
    wire_binary: np.ndarray, fork_pad: int,
    _split_continuous_segments
) -> bool:
    h, w = wire_binary.shape
    x1, y1 = max(0, x - fork_pad), max(0, y - fork_pad)
    x2, y2 = min(w - 1, x + fork_pad), min(h - 1, y + fork_pad)
    roi = wire_binary[y1:y2+1, x1:x2+1]
    if roi.size == 0:
        return False
    cy, cx = y - y1, x - x1
    if roi[cy, cx] == 0:
        return False
    row = roi[cy, :]
    row_segs = _split_continuous_segments(np.where(row > 0)[0].tolist())
    has_h = any(cx in seg and len(seg) > fork_pad * 0.6 for seg in row_segs)
    col = roi[:, cx]
    col_segs = _split_continuous_segments(np.where(col > 0)[0].tolist())
    has_v = any(cy in seg and len(seg) > fork_pad * 0.6 for seg in col_segs)
    return has_h and has_v


def build_pin_start_masks(
    h: int, w: int,
    pos_to_pins: Dict[Tuple[int, int], List[Dict[str, Any]]],
    start_pt_set: Set[Tuple[int, int]],
    near_win: int
) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int,int], List[Dict[str,Any]]]]:
    """
    构建邻近掩码，替代每次双重for扫描窗口
    返回：pin_near_mask, start_near_mask, pos_to_pins（原样传回）
    mask: 255=该像素在某个pin/start_point的near_win邻域内
    """
    # 1. 基础二值图打点
    pin_dot = np.zeros((h, w), dtype=np.uint8)
    for (x, y) in pos_to_pins.keys():
        if 0 <= y < h and 0 <= x < w:
            pin_dot[y, x] = 255

    start_dot = np.zeros((h, w), dtype=np.uint8)
    for (x, y) in start_pt_set:
        if 0 <= y < h and 0 <= x < w:
            start_dot[y, x] = 255

    # 结构元：near_win × near_win 正方形，等价你原来窗口扫描
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (near_win, near_win))
    pin_near_mask = cv2.dilate(pin_dot, kernel, iterations=1)
    start_near_mask = cv2.dilate(start_dot, kernel, iterations=1)
    return pin_near_mask, start_near_mask, pos_to_pins


def build_pixel2pin(
    h: int, w: int,
    pos_to_pins: Dict[Tuple[int, int], List[Dict[str, Any]]],
    near_win: int
) -> Dict[Tuple[int,int], List[Tuple[int,int]]]:
    """
    pixel2pin[(x,y)] = 窗口内所有pin中心点列表
    只给邻域内像素建立映射，用于mask命中后取回pin坐标
    """
    pixel2pin = {}
    half = near_win // 2
    for (px, py), pinlist in pos_to_pins.items():
        x0 = max(0, px - half)
        x1 = min(w, px - half + near_win)
        y0 = max(0, py - half)
        y1 = min(h, py - half + near_win)
        for y in range(y0, y1):
            for x in range(x0, x1):
                pixel2pin.setdefault((x,y), []).append((px,py))
    return pixel2pin


def find_pin_near_fast(
    x: int, y: int,
    pin_near_mask: np.ndarray,
    pixel2pin: Dict[Tuple[int,int], List[Tuple[int,int]]]
) -> Tuple[int, int] | None:
    if pin_near_mask[y, x] == 0:
        return None
    lst = pixel2pin.get((x,y), [])
    if len(lst) > 0:
        return lst[0]
    return None


def has_start_near_with_dir(
    x: int, y: int,
    dx: int, dy: int,
    start_near_mask: np.ndarray,
    start_pt_set: Set[Tuple[int, int]],
    angle_thresh_deg: float = 90.0
) -> bool:
    """
    条件：
    1. (x,y) 在 start_near_mask 邻域内
    2. 当前前进方向(dx,dy) 朝向最近的start中心点
    angle_thresh_deg：夹角阈值，<=90°代表方向是朝着目标点（相向）
    """
    if not (0 <= x < start_near_mask.shape[1] and 0 <= y < start_near_mask.shape[0]):
        return False
    if start_near_mask[y, x] <= 0:
        return False

    # 找到这个邻域内距离当前(x,y)最近的start中心点
    min_dist = float("inf")
    best_pt = None
    for (sx, sy) in start_pt_set:
        d = np.hypot(sx - x, sy - y)
        if d < min_dist:
            min_dist = d
            best_pt = (sx, sy)
    if best_pt is None:
        return False

    # 向量 v：从当前像素指向start中心
    vx = best_pt[0] - x
    vy = best_pt[1] - y
    # 走线前进方向向量 u
    ux = dx
    uy = dy

    # 点积求夹角
    dot = vx * ux + vy * uy
    mod_v = np.hypot(vx, vy)
    mod_u = np.hypot(ux, uy)
    if mod_v < 1e-6 or mod_u < 1e-6:
        return True
    cos_theta = dot / (mod_v * mod_u)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_deg = np.rad2deg(np.arccos(cos_theta))

    return theta_deg <= angle_thresh_deg


def clear_orth_wire(
    x: int, y: int, dx: int, dy: int,
    walkable: np.ndarray, h: int, w: int
):
    if dx != 0:
        y_up = y
        while y_up > 0 and walkable[y_up - 1, x] > 0:
            y_up -= 1
        y_down = y
        while y_down < h - 1 and walkable[y_down + 1, x] > 0:
            y_down += 1
        walkable[y_up:y_down + 1, x] = 0
    else:
        x_left = x
        while x_left > 0 and walkable[y, x_left - 1] > 0:
            x_left -= 1
        x_right = x
        while x_right < w - 1 and walkable[y, x_right + 1] > 0:
            x_right += 1
        walkable[y, x_left:x_right + 1] = 0


def is_near_stop(visited_stop, x, y, near_win):
    fork_dup = False
    for (ox, oy) in visited_stop:
        if math.hypot(x - ox, y - oy) < near_win:
            fork_dup = True
            break
    return fork_dup


def orth_fork_directions(
    x: int, y: int,
    wire_binary: np.ndarray,
    scan_range: int,
    thres: float
) -> set[tuple[int, int]]:
    """
    检测点(x,y)四个正方向：right(1,0), left(-1,0), bottom(0,1), top(0,-1)
    返回 (是否存在有效方向, 有效方向集合)
    """
    h, w = wire_binary.shape
    four_dirs = [
        (1, 0),   # right
        (-1, 0),  # left
        (0, 1),   # bottom
        (0, -1)   # top
    ]
    valid_dirs = set()

    for odx, ody in four_dirs:
        if odx != 0:
            # 水平方向，扫描同一行
            fixed_y = y
            if odx > 0:
                x_low = x
                x_high = min(w - 1, x + scan_range)
            else:
                x_low = max(0, x - scan_range)
                x_high = x
            if x_low > x_high:
                continue
            row = wire_binary[fixed_y, x_low:x_high+1]
            total = x_high - x_low + 1
            if row.sum() > total * thres:
                valid_dirs.add((odx, ody))
        else:
            # 竖直方向，扫描同一列
            fixed_x = x
            if ody > 0:
                y_low = y
                y_high = min(h - 1, y + scan_range)
            else:
                y_low = max(0, y - scan_range)
                y_high = y
            if y_low > y_high:
                continue
            col = wire_binary[y_low:y_high+1, fixed_x]
            total = y_high - y_low + 1
            if col.sum() > total * thres:
                valid_dirs.add((odx, ody))
    return valid_dirs


def trace_single_net_dfs(
    start_pin: Dict[str, Any],
    walkable: np.ndarray,
    wire_binary: np.ndarray,
    intersect: np.ndarray,
    bgr_img: np.ndarray,          # HSV H通道
    diff_thresh: int,       # 和extract一致的色相阈值
    comp_mask: np.ndarray,
    pos_to_pins: Dict[Tuple[int, int], List[Dict[str, Any]]],
    pixel2pin: Dict[Tuple[int,int], List[Tuple[int,int]]],
    pin_near_mask: np.ndarray,
    start_near_mask: np.ndarray,
    start_pt_set: Set[Tuple[int, int]],
    visited_pins: Set[int],
    visited_stop: Set[Tuple[int,int]],
    scan_range: int,
    thres: float,
    stop_near_win: int,
    DIRS,
    fly_back_step = 3
) -> Tuple[Set[str], List[List[Dict[str, int]]]]:
    h, w = wire_binary.shape
    net_pin_ids = set()
    net_edge_list = []

    sx, sy = start_pin["x"], start_pin["y"]
    start_key = (sx, sy)
    for p in pos_to_pins.get(start_key, []):
        net_pin_ids.add(p["full_id"])
        visited_pins.add(p["gid"])
    visited_stop.add(start_key)

    stack = []
    for dx, dy in DIRS:
        nx, ny = sx + dx, sy + dy
        if 0 <= nx < w and 0 <= ny < h:
            if walkable[ny, nx] > 0 and comp_mask[ny, nx] == 0:
                stack.append((nx, ny, dx, dy, sx, sy))

    # print(start_pin)

    while stack:
        cx, cy, dx, dy, last_kx, last_ky = stack.pop()
        curr_x, curr_y = cx, cy
        fork_points = []
        max_num_fork = 0
        in_fork = False
        stop_type = None
        stop_x, stop_y = curr_x, curr_y

        while True:
            print(f" -> ({curr_x}, {curr_y})")
            # if 1250 < curr_x < 1300 and 450 < curr_y < 500:
            #     import pdb; pdb.set_trace()

            cond = not (0 <= curr_x < w and 0 <= curr_y < h) or has_start_near_with_dir(curr_x, curr_y, dx, dy, start_near_mask, start_pt_set, angle_thresh_deg=90.0)
            fly_x = curr_x - dx * fly_back_step
            fly_y = curr_y - dy * fly_back_step
            if not cond and not is_near_stop(visited_stop, curr_x, curr_y, stop_near_win * 3) and walkable[curr_y, curr_x] == 0 and not in_fork and intersect[fly_y, fly_x] == 1:
                stop_type = "fly"
                stop_x = max(0, min(w - 1, fly_x))
                stop_y = max(0, min(h - 1, fly_y))
                break

            if cond or (walkable[curr_y, curr_x] == 0 and not in_fork):
                stop_type = "dead"
                break

            is_fork, num_forks = orth_fork_check(curr_x, curr_y, dx, dy, wire_binary, bgr_img, scan_range, thres, diff_thresh)
            max_num_fork = max(max_num_fork, num_forks)
            if is_fork:
                if not in_fork:
                    in_fork = True
                fork_points.append((curr_x, curr_y, num_forks))
            else:
                if in_fork:
                    record_max_num_fork = max([p[2] for p in fork_points])
                    confirm_fork = False
                    if record_max_num_fork == max_num_fork:
                        fork_candidates = [p for p in fork_points if p[2] == max_num_fork]
                        fork_cx, fork_cy, _ = fork_candidates[len(fork_candidates) // 2]
                        near_stops = [
                            (sx, sy) for (sx, sy) in visited_stop
                            if np.hypot(fork_cx - sx, fork_cy - sy) <= stop_near_win
                        ]
                    
                        if len(near_stops) == 0:
                            confirm_fork = True
                        else:
                            new_fork_candidates = []
                            for f in fork_candidates:
                                fork_cx, fork_cy, _ = f
                                fork_dirs = orth_fork_directions(fork_cx, fork_cy, wire_binary, scan_range, thres)
                                fork_cand = True
                                for (sx, sy) in near_stops:
                                    stop_dirs = orth_fork_directions(sx, sy, wire_binary, scan_range, thres)
                                    if len(fork_dirs - stop_dirs) == 0:
                                        fork_cand = False
                                        break
                                if fork_cand:
                                    new_fork_candidates.append(f)

                            if len(new_fork_candidates) > 0:
                                fork_cx, fork_cy, _ = new_fork_candidates[len(new_fork_candidates) // 2]
                                confirm_fork = True

                        if confirm_fork:
                            stop_x, stop_y = fork_cx, fork_cy
                            stop_type = "fork"
                            break          

                    in_fork = False
                    fork_points.clear()

                if num_forks <= 1:
                    max_num_fork = 0

            pin_pos = find_pin_near_fast(curr_x, curr_y, pin_near_mask, pixel2pin)
            if not in_fork and pin_pos is not None and pin_pos not in visited_stop:
                stop_x, stop_y = pin_pos
                stop_type = "pin"
                break

            curr_x += dx
            curr_y += dy
        # end line trace

        if stop_type == "dead":
            continue

        stop_key = (stop_x, stop_y)
        last_key = (last_kx, last_ky)
        # print(f"{last_key} -> {stop_key} | {stop_type}")

        if stop_key in visited_stop:
            continue
        visited_stop.add(stop_key)

        not_near_pins = not (last_key in pos_to_pins and stop_key in pos_to_pins and math.hypot(last_kx - stop_x, last_ky - stop_y) < stop_near_win)
        if last_key != stop_key and not_near_pins:
            net_edge_list.append([
                {"x": last_kx, "y": last_ky},
                {"x": stop_x, "y": stop_y}
            ])

        if stop_type == 'fly':
            continue

        if stop_type == "pin":
            for p in pos_to_pins[stop_key]:
                if p["gid"] not in visited_pins:
                    net_pin_ids.add(p["full_id"])
                    visited_pins.add(p["gid"])

        for ndx, ndy in DIRS:
            if (ndx, ndy) == (-dx, -dy):
                continue
            nx, ny = stop_x + ndx, stop_y + ndy
            if 0 <= nx < w and 0 <= ny < h:
                if walkable[ny, nx] > 0 and intersect[ny, nx] == 1:
                    stack.append((nx, ny, ndx, ndy, stop_x, stop_y))

    return net_pin_ids, net_edge_list


def build_pcb_nets(
    comp_pins: Dict[int, List[Dict[str, Any]]],
    wire_binary: np.ndarray,
    intersect: np.ndarray,
    bgr_img: np.ndarray,    # 新增，用来生成H
    comps: List[Any],
    comp_start_pts: Dict[int, List[Tuple[int, int, str]]],
    scan_range: int = 10,
    thres: float = 0.9,
    near_win: int = 5,
    stop_near_win: int = 5,
    diff_thresh: int = 20  # 和extract_comp_pins_straight_hsv默认对齐
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    h, w = wire_binary.shape
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0]

    meet_pins, normal_pins, pos_to_pins, pins_dict = prepare_pin_data(comp_pins)
    nets = build_meet_pin_nets(meet_pins)
    comp_mask, start_pt_set = build_obstacle_mask(h, w, comps, comp_start_pts)

    pin_near_mask, start_near_mask, pos_to_pins = build_pin_start_masks(
        h, w, pos_to_pins, start_pt_set, near_win*2
    )
    pixel2pin = build_pixel2pin(h, w, pos_to_pins, near_win*2)

    walkable = wire_binary.copy()
    visited_pins: Set[int] = set()
    visited_stop: Set[Tuple[int,int]] = set()
    for start_pin in tqdm(normal_pins):
        if start_pin["gid"] in visited_pins:
            continue
        net_pin_ids, net_edge_list = trace_single_net_dfs(
            start_pin=start_pin,
            walkable=walkable,
            wire_binary=wire_binary,
            intersect=intersect,
            bgr_img=bgr_img,
            diff_thresh=diff_thresh,
            comp_mask=comp_mask,
            pos_to_pins=pos_to_pins,
            pixel2pin=pixel2pin,
            pin_near_mask=pin_near_mask,
            start_near_mask=start_near_mask,
            start_pt_set=start_pt_set,
            visited_pins=visited_pins,
            visited_stop=visited_stop,
            scan_range=scan_range,
            thres=thres,
            stop_near_win=stop_near_win,
            DIRS=DIRS,
        )
        edges_dict = {}
        for idx, edge in enumerate(net_edge_list):
            edges_dict[f"edge_{idx}"] = edge
        nets.append({
            "hyperGraph": f"({','.join(sorted(net_pin_ids))})",
            "edges": edges_dict
        })

    nets_dict = {}
    for idx, net in enumerate(nets):
        nets_dict[f"net_{idx}"] = net
    return pins_dict, nets_dict


def visualize_start_points_with_pins(
    wire_binary: np.ndarray,
    comp_start_pts: Dict[int, List[Tuple[int, int, str]]],
    comp_pins: Dict[int, List[Dict[str, Any]]],
    nets_dict: Dict[str, Dict[str, Any]],
    case_name: str = "your_case_name",
    out_root: str = "outputs/viz"
):
    import random
    h, w = wire_binary.shape
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    canvas[wire_binary > 0] = (220, 220, 220)

    BLUE = (255, 0, 0)   # BGR OpenCV蓝
    RED = (0, 0, 255)    # BGR OpenCV红

    def random_edge_color() -> Tuple[int, int, int]:
        """
        随机生成BGR颜色，避开纯红、纯蓝，
        保证和start_pt(蓝)、pin(红)区分开
        约束：
        - 不能接近 (255,0,0) 蓝色
        - 不能接近 (0,0,255) 红色
        """
        while True:
            b = random.randint(0, 255)
            g = random.randint(0, 255)
            r = random.randint(0, 255)
            # 距离阈值，过滤靠近红蓝的颜色
            dist_blue = ((b - 255)**2 + g*g + r*r)
            dist_red = (b*b + g*g + (r - 255)**2)
            if dist_blue > 12000 and dist_red > 12000:
                return (b, g, r)

    # 给每个net预先分配一个固定随机色，同一个net所有edge同色
    net_color_map: Dict[str, Tuple[int, int, int]] = {}
    for net_key in nets_dict.keys():
        net_color_map[net_key] = random_edge_color()

    # ---- 1. 先画 nets edges ----
    for net_key, net_data in nets_dict.items():
        c = net_color_map[net_key]
        edges = net_data.get("edges", {})
        for edge_key, pts_list in edges.items():
            if len(pts_list) < 2:
                continue
            p0, p1 = pts_list[0], pts_list[1]
            x0, y0 = int(p0["x"]), int(p0["y"])
            x1, y1 = int(p1["x"]), int(p1["y"])
            cv2.line(canvas, (x0, y0), (x1, y1), color=c, thickness=2)

    # ---- 2. comp_start_pts 画蓝色 ----
    for cid, pts in comp_start_pts.items():
        for (x, y, side) in pts:
            cv2.circle(canvas, (int(x), int(y)), 1, BLUE, -1)

    # ---- 3. comp_pins 画红色 ----
    for cid, pin_list in comp_pins.items():
        for pin in pin_list:
            x, y = int(pin["x"]), int(pin["y"])
            cv2.circle(canvas, (x, y), 2, RED, -1)
            t = pin.get("type", "")
            if t:
                cv2.putText(canvas, t[:3], (x + 6, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, RED, 1)

    save_dir = os.path.join(out_root, case_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "viz_net_pin.png")
    cv2.imwrite(save_path, canvas)
    print(f"[viz] saved -> {save_path}")


def extract_nets(
    img: np.ndarray,
    comps: List[Any],
    texts: List[Any],
    img_path : str,
    visualize: bool = True,
    viz_root: str = "outputs/viz",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    img_name = Path(img_path).parent.name
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    binary_raw, final_t = _get_wire_binary(img, gray, comps)
    texts_cover_mask = _get_cross_mask((h,w), texts, is_cross=True, border_px=TEXT_CM_BORDER_PX)
    comps_cover_mask = _get_cross_mask((h,w), comps, is_cross=False, border_px=COMP_CM_BORDER_PX)
    all_cover_mask = cv2.bitwise_and(texts_cover_mask, comps_cover_mask)
    
    comp_start_pts = _get_comp_start_points(
        cv2.bitwise_and(binary_raw, all_cover_mask),
        comps,
        step_cfg=CSP_STEP_CFG,
        edge_pad_px=CSP_EDGE_PAD_PX
    )

    comps, comp_start_pts = refine_comp_and_start_ps(
        wire_binary=binary_raw,
        comps=comps,
        comp_start_pts=comp_start_pts
    )
    comps_cover_mask = _get_cross_mask((h,w), comps, is_cross=False, border_px=COMP_CM_BORDER_PX)
    all_cover_mask = cv2.bitwise_and(texts_cover_mask, comps_cover_mask)

    # comp_pins, delta_mat = extract_comp_pins_straight_hsv(
    #     bgr_img=bgr,
    #     comp_start_pts=comp_start_pts,
    #     wire_binary=binary_raw,
    #     diff_thresh=EP_DIFF_THRESH
    # )

    comp_pins, delta_mat = extract_comp_pins_straight_hsv(
        bgr_img=bgr,
        comp_start_pts=comp_start_pts,
        wire_binary=cv2.bitwise_and(binary_raw, all_cover_mask),
        scan_range=RCP_SCAN_RANGE,
        delta_dev_thresh=EP_DIFF_THRESH,
        near_start_thresh=BN_NEAR_WIN
    )
    # plot_delta_heatmap(bgr, delta_mat, diff_thresh=EP_DIFF_THRESH, save_path=f"outputs/viz/{img_name}/delta_heat.png")
    comp_pins = refine_and_cluster_pins(
        comp_pins=comp_pins,
        comp_start_pts=comp_start_pts,
        wire_binary=binary_raw,
        max_back_step=RCP_MAX_BACK_STEP
    )

    # comp_pins = fix_pin_by_llm(binary_raw, img, comps, comp_pins, img_name)
    pins_dict, nets_dict = build_pcb_nets(
        comp_pins=comp_pins,
        wire_binary=binary_raw,
        intersect=cv2.bitwise_and(binary_raw, all_cover_mask),
        bgr_img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        comps=comps,
        comp_start_pts=comp_start_pts,
        scan_range=BN_SCAN_RANGE,
        thres=BN_THRES,
        diff_thresh=BN_DIFF_THRESH,
        near_win=BN_NEAR_WIN,
        stop_near_win=BN_STOP_NEAR_WIN
    )

    # visualize_start_points_with_pins(
    #     wire_binary=cv2.bitwise_and(binary_raw, all_cover_mask),
    #     comp_start_pts=comp_start_pts,
    #     comp_pins=comp_pins,
    #     nets_dict=nets_dict,
    #     case_name=img_name
    # )

    return pins_dict, nets_dict
