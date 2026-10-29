import cv2
import numpy as np
import heapq
import os
from skimage.morphology import skeletonize
from collections import deque
from typing import List, Tuple, Dict, Any, Set
from tqdm import tqdm
import math
from pathlib import Path
from sklearn.cluster import KMeans

# ===================== PCB走线提取配置参数区（统一管理） =====================
# _get_wire_binary
WB_OFFSET = 5

# _get_cross_mask
CM_SCALE = 0.8
CM_CROSS_WIDTH = 2

# _get_comp_start_points
CSP_STEP_CFG = {
    "default": (-2, 15),
    "ground": (0, 15),
    "inductor": (-1, 5)
}
CSP_BOUNDARY_RATIO_THRESH = 0.9

# extract_comp_pins_straight_hsv
EP_DIFF_THRESH = 20

# refine_and_cluster_pins
RCP_SCAN_RANGE = 10
RCP_MAX_BACK_STEP = 10

# build_pcb_nets
BN_SCAN_RANGE = 15
BN_THRES = 0.99
BN_FORK_PAD = 5
BN_DIFF_THRESH = 30
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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

    # ========== 构造 comp_binary ==========
    # comp_binary = np.zeros_like(binary_raw)
    # # 第一步：生成所有器件bbox的联合mask
    # comp_mask = np.zeros_like(binary_raw, dtype=bool)
    # for c in comps:
    #     x1, y1, x2, y2 = c.bbox
    #     x1 = max(0, int(x1))
    #     y1 = max(0, int(y1))
    #     x2 = min(w, int(x2))
    #     y2 = min(h, int(y2))
    #     comp_mask[y1:y2, x1:x2] = True

    # # 感兴趣区域：同时满足【前景】AND【在器件bbox内】
    # roi_mask = (binary_raw == 1) & comp_mask
    # roi_y, roi_x = np.where(roi_mask)
    # if len(roi_y) < 2:
    #     # 像素太少无法聚类，comp_binary直接全0
    #     wire_binary = binary_raw.copy()
    #     return binary_raw, comp_binary, wire_binary, final_t

    # # 取出BGR彩色像素做聚类
    # roi_pixels = img[roi_y, roi_x, :].astype(np.float32)
    # kmeans = KMeans(n_clusters=2, n_init="auto", random_state=42)
    # labels = kmeans.fit_predict(roi_pixels)

    # # 统计两类数量
    # cnt0 = np.sum(labels == 0)
    # cnt1 = np.sum(labels == 1)
    # # 保留像素多的一类作为器件主体，少的一类（走线）置0
    # keep_label = 0 if cnt0 > cnt1 else 1

    # # 回填 comp_binary
    # keep_idx = labels == keep_label
    # comp_binary[roi_y[keep_idx], roi_x[keep_idx]] = 1

    # # wire_binary = binary_raw 减去 comp_binary
    # wire_binary = np.where((binary_raw == 1) & (comp_binary == 0), 1, 0).astype(np.uint8)
    comp_binary, wire_binary = np.array([]), np.array([])
    return binary_raw, comp_binary, wire_binary, final_t


def _get_cross_mask(
    img_shape: Tuple[int, int],
    texts: List[Any],
    comps: List[Any],
    use_cross: bool,    # True：十字消除；False：整个缩框全消除
    scale=0.9,
    cross_width=2
) -> np.ndarray:
    """
    mask规则：
    1. 原图mask默认 255
    2. 每个text bbox：先等比例缩放到 scale
       - use_cross=True：缩框内部不整体置0，只在缩后bbox中心画十字，十字区域置0
       - use_cross=False：缩框内部全部置0（全消除）
    3. comps暂时先传入预留，如果你后续要对器件框做同样逻辑可以扩展；当前版本仅text生效
    """
    h, w = img_shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    half_cross = cross_width // 2

    for ele in texts + comps:
        x1, y1, x2, y2 = map(int, ele.bbox)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = x2 - x1
        bh = y2 - y1

        new_bw = bw * scale
        new_bh = bh * scale
        sx1 = cx - new_bw / 2
        sy1 = cy - new_bh / 2
        sx2 = cx + new_bw / 2
        sy2 = cy + new_bh / 2

        sx1, sy1 = max(0, int(sx1)), max(0, int(sy1))
        sx2, sy2 = min(w, int(sx2)), min(h, int(sy2))

        if not use_cross:
            # 全消除模式：缩框整块置0
            mask[sy1:sy2, sx1:sx2] = 0
        else:
            # 十字消除模式：只画十字，框本体保留255
            cx_int = int(round(cx))
            # 水平十字线：横向穿过中心，y范围为原始text bbox y1~y2
            x_c1 = max(0, cx_int - half_cross)
            x_c2 = min(w, cx_int + half_cross + 1)
            mask[y1:y2, x_c1:x_c2] = 0

            cy_int = int(round(cy))
            # 垂直十字线：纵向穿过中心，x范围为原始text bbox x1~x2
            y_c1 = max(0, cy_int - half_cross)
            y_c2 = min(h, cy_int + half_cross + 1)
            mask[y_c1:y_c2, x1:x2] = 0
    return mask



def visualize_pcb_debug(gray: np.ndarray, wire_binary: np.ndarray, skel: np.ndarray, intersect: np.ndarray, case_name: str, out_root="debug"):
    os.makedirs(out_root, exist_ok=True)
    bin255 = (wire_binary * 255).astype(np.uint8)
    skel255 = (skel * 255).astype(np.uint8)
    inter255 = (intersect * 255).astype(np.uint8)
    canvas = np.hstack([gray, bin255, skel255, inter255])
    out_path = os.path.join(out_root, f"{case_name}_debug.png")
    cv2.imwrite(out_path, canvas)
    return out_path


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
    boundary_ratio_thresh: float = 0.9,
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
        ratio = np.sum(col > 0) / len(col)
        if ratio > boundary_ratio_thresh:
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
    boundary_ratio_thresh: float = 0.9,
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
        ratio = np.sum(row > 0) / len(row)
        if ratio > boundary_ratio_thresh:
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


def _get_comp_start_points(
    wire_binary: np.ndarray,
    comps: List[Any],
    step_cfg: Dict[str, Tuple[int, int]],
    boundary_ratio_thresh: float = 0.9,
) -> Dict[int, List[Tuple[int, int, str]]]:
    start_points = {}
    for comp_idx, comp in enumerate(comps):
        start_step, total_steps = step_cfg.get(comp.label, step_cfg['default'])
        x1, y1, x2, y2 = map(int, comp.bbox)
        pts = []
        pts.extend(_process_horizontal_edge(wire_binary, x2, y1, y2, start_step, total_steps, dx=1, side="right", boundary_ratio_thresh=boundary_ratio_thresh))
        pts.extend(_process_horizontal_edge(wire_binary, x1, y1, y2, start_step, total_steps, dx=-1, side="left", boundary_ratio_thresh=boundary_ratio_thresh))
        pts.extend(_process_vertical_edge(wire_binary, y2, x1, x2, start_step, total_steps, dy=1, side="bottom", boundary_ratio_thresh=boundary_ratio_thresh))
        pts.extend(_process_vertical_edge(wire_binary, y1, x1, x2, start_step, total_steps, dy=-1, side="top", boundary_ratio_thresh=boundary_ratio_thresh))
        start_points[comp_idx] = pts
    return start_points


def extract_comp_pins_straight_hsv(
    gray: np.ndarray,
    bgr_img: np.ndarray,
    comp_start_pts: Dict[int, List[Tuple[int, int, str]]],
    wire_binary: np.ndarray,
    diff_thresh: int = 20,
    max_step: int = 2000
) -> Dict[int, List[Dict[str, Any]]]:
    """仅沿初始方向直线追踪，不拐弯，HSV色相环形差值判定
    修复：双向起点相遇时两边都停在meet中点，不会一侧变成dead_end；meet pin补齐side字段
    """
    h, w = gray.shape
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0]
    side2dir = {"right": (1, 0), "left": (-1, 0), "bottom": (0, 1), "top": (0, -1)}

    # map: (起点x,y) -> (器件cid, side)
    start_point_map: Dict[Tuple[int, int], Tuple[int, str]] = {}
    for cid, pts in comp_start_pts.items():
        for x, y, side in pts:
            start_point_map[(x, y)] = (cid, side)

    comp_pins: Dict[int, List[Dict[str, Any]]] = {cid: [] for cid in comp_start_pts.keys()}
    # 记录【已经配对完成的起点对】，防止其中一方继续追踪跑到dead_end
    handled_pairs = set()
    win_radius = 1

    for comp_idx, start_pts in comp_start_pts.items():
        for (sx, sy, side) in start_pts:
            dx, dy = side2dir[side]
            cx, cy = sx, sy
            path: List[Tuple[int, int]] = []
            step = 0

            while 0 <= cx < w and 0 <= cy < h and step < max_step:
                if wire_binary[cy, cx] <= 0:
                    break
                path.append((cx, cy))
                step += 1

                # HSV色相跳变检测（gradient pin，原有逻辑不变）
                if len(path) >= 7:
                    x_pre, y_pre = path[-5]
                    x1p, x2p = max(0, x_pre - win_radius), min(w - 1, x_pre + win_radius)
                    y1p, y2p = max(0, y_pre - win_radius), min(h - 1, y_pre + win_radius)
                    h_pre_avg = H[y1p:y2p+1, x1p:x2p+1].mean()

                    x_post, y_post = path[-3]
                    x1po, x2po = max(0, x_post - win_radius), min(w - 1, x_post + win_radius)
                    y1po, y2po = max(0, y_post - win_radius), min(h - 1, y_post + win_radius)
                    h_post_avg = H[y1po:y2po+1, x1po:x2po+1].mean()

                    delta = abs(h_pre_avg - h_post_avg)
                    delta = min(delta, 180 - delta)
                    if delta > diff_thresh:
                        comp_pins[comp_idx].append({"x": cx, "y": cy, "type": "gradient", "side": side})
                        break

                # ========== 相遇逻辑修复 ==========
                if (cx, cy) in start_point_map:
                    other_cid, other_side = start_point_map[(cx, cy)]
                    if other_cid != comp_idx:
                        # 构造起点标识对，无序key，A-B 和 B-A视为同一对
                        key = frozenset({(sx, sy, comp_idx, side), (cx, cy, other_cid, other_side)})
                        if key not in handled_pairs:
                            handled_pairs.add(key)
                            mid_idx = len(path) // 2
                            mx, my = path[mid_idx]
                            # 两个pin都带上各自的side！refine需要side
                            comp_pins[comp_idx].append({"x": mx, "y": my, "type": "meet", "side": side})
                            comp_pins[other_cid].append({"x": mx, "y": my, "type": "meet", "side": other_side})
                        break

                # 向前走一步
                nx = cx + dx
                ny = cy + dy
                if not (0 <= nx < w and 0 <= ny < h and wire_binary[ny, nx] > 0):
                    comp_pins[comp_idx].append({"x": cx, "y": cy, "type": "dead_end", "side": side})
                    break
                cx, cy = nx, ny
            # end while
    return comp_pins


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
    scan_range: int = 30,
    max_back_step: int = 120,
    thres: float = 0.9
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


# ===================== 底层工具函数（独立出来） =====================
def orth_fork_check(
    x: int, y: int, dx: int, dy: int,
    wire_binary: np.ndarray,
    H: np.ndarray,
    scan_range: int,
    thres: float,
    diff_thresh: int,
    block_len: int = 3
) -> tuple[bool, int]:
    h, w = wire_binary.shape

    def h_delta(a, b):
        d = abs(a - b)
        return min(d, 180 - d)

    def sample_block_h(cx, cy, step_dx, step_dy):
        """沿 (step_dx, step_dy) 取 block_len 个连续像素，返回H均值；越界像素丢弃"""
        vals = []
        for s in range(block_len):
            nx = cx + step_dx * s
            ny = cy + step_dy * s
            if 0 <= nx < w and 0 <= ny < h:
                vals.append(H[ny, nx])
        if not vals:
            return None
        return float(np.mean(vals))

    # 前进、左、右 正交方向
    fd_dir = (dx, dy)
    left_dir = (-dy, dx)
    right_dir = (dy, -dx)
    dirs = [fd_dir, left_dir, right_dir]
    ok = []
    end_pos = []  # 每个方向scan_range远端坐标

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
        end_pos.append((ex, ey, ddx, ddy))

    thres_cnt = sum(ok)
    fd_ok, left_ok, right_ok = ok
    fd_end, left_end, right_end = end_pos
    
    # ========= 规则分支 =========
    if fd_ok and left_ok and right_ok:
        # 三者全部满足，进入颜色校验
        h_curr = sample_block_h(x, y, 0, 0)
        if h_curr is None:
            return False, thres_cnt
        # 三个方向远端各取1*3色块
        h_fd = sample_block_h(fd_end[0], fd_end[1], fd_dir[0], fd_dir[1])
        h_left = sample_block_h(left_end[0], left_end[1], left_dir[0], left_dir[1])
        h_right = sample_block_h(right_end[0], right_end[1], right_dir[0], right_dir[1])
        for hv in [h_fd, h_left, h_right]:
            if hv is not None and h_delta(h_curr, hv) < diff_thresh:
                return False, thres_cnt
        return True, thres_cnt
    else:
        # 三者不同时满足：左右任意ok就True，都不ok False
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


def has_start_near_fast(
    x: int, y: int,
    start_near_mask: np.ndarray
) -> bool:
    return bool(start_near_mask[y, x] > 0)


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
    H: np.ndarray,          # HSV H通道
    diff_thresh: int,       # 和extract一致的色相阈值
    comp_mask: np.ndarray,
    pos_to_pins: Dict[Tuple[int, int], List[Dict[str, Any]]],
    pixel2pin: Dict[Tuple[int,int], List[Tuple[int,int]]],
    pin_near_mask: np.ndarray,
    start_near_mask: np.ndarray,
    visited_pins: Set[int],
    visited_stop: Set[Tuple[int,int]],
    scan_range: int,
    thres: float,
    fork_pad: int,
    stop_near_win: int,
    DIRS,
    _split_continuous_segments,
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
        # if stack[-1] == (1211, 497, 0, -1, 1211, 498):
        #     import pdb; pdb.set_trace
        cx, cy, dx, dy, last_kx, last_ky = stack.pop()
        curr_x, curr_y = cx, cy
        fork_points = []
        max_num_fork = 0
        in_fork = False
        stop_type = None
        stop_x, stop_y = curr_x, curr_y
        
        while True:
            print(f" -> ({curr_x}, {curr_y})")
            # if 1250 < curr_x < 1300 and 100 < curr_y < 150:
            #     import pdb; pdb.set_trace()
            
            cond = not (0 <= curr_x < w and 0 <= curr_y < h) or has_start_near_fast(curr_x, curr_y, start_near_mask)
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

            is_fork, num_forks = orth_fork_check(curr_x, curr_y, dx, dy, wire_binary, H, scan_range, thres, diff_thresh)
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

                        
            # if not in_fork:
            #     clear_orth_wire(curr_x, curr_y, dx, dy, walkable, h, w)
            curr_x += dx
            curr_y += dy
        # end line trace

        if stop_type == "dead":
            continue
# (Pdb) stop_key
# (1258, 121)
# (Pdb) pin_pos
# (1258, 119)
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
                if walkable[ny, nx] > 0 and comp_mask[ny, nx] == 0:
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
    fork_pad: int = 5,
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
            H=H,
            diff_thresh=diff_thresh,
            comp_mask=comp_mask,
            pos_to_pins=pos_to_pins,
            pixel2pin=pixel2pin,
            pin_near_mask=pin_near_mask,
            start_near_mask=start_near_mask,
            visited_pins=visited_pins,
            visited_stop=visited_stop,
            scan_range=scan_range,
            thres=thres,
            fork_pad=fork_pad,
            stop_near_win=stop_near_win,
            DIRS=DIRS,
            _split_continuous_segments=_split_continuous_segments
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


def visualize_binary(img_name, binary_raw, comp_binary, wire_binary):
    viz_dir = Path("outputs/viz") / img_name
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 0/1 -> 0/255
    cv2.imwrite(str(viz_dir / "binary_raw.png"), binary_raw * 255)
    cv2.imwrite(str(viz_dir / "comp_binary.png"), comp_binary * 255)
    cv2.imwrite(str(viz_dir / "wire_binary.png"), wire_binary * 255)


def extract_nets(
    img: np.ndarray,
    comps: List[Any],
    texts: List[Any],
    img_path : str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    img_name = Path(img_path).parent.name
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    binary_raw, comp_binary, wire_binary, _ = _get_wire_binary(img, gray, comps, offset=WB_OFFSET)
    # visualize_binary(img_name, binary_raw, comp_binary, wire_binary)

    cover_mask = _get_cross_mask((h,w), texts, comps, use_cross=False, scale=CM_SCALE, cross_width=CM_CROSS_WIDTH)
    intersect = cv2.bitwise_and(binary_raw, cover_mask)

    comp_start_pts = _get_comp_start_points(
        binary_raw,
        comps,
        step_cfg=CSP_STEP_CFG,
        boundary_ratio_thresh=CSP_BOUNDARY_RATIO_THRESH
    )
    comp_pins = extract_comp_pins_straight_hsv(
        gray=gray,
        bgr_img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        comp_start_pts=comp_start_pts,
        wire_binary=binary_raw,
        diff_thresh=EP_DIFF_THRESH
    )
    comp_pins = refine_and_cluster_pins(
        comp_pins=comp_pins,
        comp_start_pts=comp_start_pts,
        wire_binary=binary_raw,
        scan_range=RCP_SCAN_RANGE,
        max_back_step=RCP_MAX_BACK_STEP
    )
    pins_dict, nets_dict = build_pcb_nets(
        comp_pins=comp_pins,
        wire_binary=binary_raw,
        intersect=intersect,
        bgr_img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        comps=comps,
        comp_start_pts=comp_start_pts,
        scan_range=BN_SCAN_RANGE,
        thres=BN_THRES,
        fork_pad=BN_FORK_PAD,
        diff_thresh=BN_DIFF_THRESH,
        near_win=BN_NEAR_WIN,
        stop_near_win=BN_STOP_NEAR_WIN
    )

    visualize_start_points_with_pins(
        wire_binary=intersect,
        comp_start_pts=comp_start_pts,
        comp_pins=comp_pins,
        nets_dict=nets_dict,
        case_name=img_name
    )

    return pins_dict, nets_dict

# TODO:
# 2. 两个非直连交叉点不能连接的问题 case10
# 3. 线段相连 case12
# 4. 出发点识别为文字部分 case13
# 5. fly 点判定精进 case13
# 6. 前后移动找中点会错误 case13