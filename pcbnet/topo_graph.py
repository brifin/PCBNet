import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
from pathlib import Path
from os.path import join
from .visual import visualize_mask, visualize_skeleton, visualize_nodes_obstacle, visualize_grid

# -------------------------- 1. 导线提取与骨架化 --------------------------
def _get_wire_mask(
    img: np.ndarray,
    diff_thresh: int = 15,
    blur_ksize: int = 7,
    exclude_red: bool = False
) -> np.ndarray:
    """
    通用非背景前景提取：自适应背景差分法
    原理：大核高斯模糊估计背景层，原图与背景做差，差值大于阈值即为非背景前景
    适配任意均匀背景色，不依赖导线颜色；支持可选过滤红色丝印
    
    Args:
        diff_thresh: 前景差异阈值，值越大越严格，背景干净可调小，背景脏可调大
        blur_ksize: 背景估计模糊核大小（奇数），越大背景越平滑，建议取图像宽的1/20
        exclude_red: 是否过滤红色区域（用于排除红色丝印文字，只保留导线）
    """
    # 1. 大核模糊估计背景层（抹平导线、文字等细节，只剩底色）
    bg = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    
    # 2. 原图与背景做差分，三通道取最大差值（提升彩色敏感度）
    diff = cv2.absdiff(img, bg)
    if len(diff.shape) == 3:
        diff = np.max(diff, axis=2)
    else:
        diff = diff
    
    # 3. 阈值分割得到非背景掩码
    mask = (diff > diff_thresh).astype(np.uint8) * 255
    
    # 4. 可选：过滤红色丝印文字（你的图里KEY、阻值等红色标注）
    if exclude_red and len(img.shape) == 3:
        b, g, r = cv2.split(img)
        # 红色判定：红通道显著高于绿蓝通道
        red_area = (r > 110) & (r - g > 20) & (r - b > 20)
        mask[red_area] = 0
    
    # 5. 形态学优化：开运算去噪点，闭运算修复导线断线
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask


def _get_skeleton(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen细化算法提取单像素宽导线骨架"""
    try:
        # 优先使用opencv-contrib的高效实现
        return cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except AttributeError:
        # 备用原生实现
        skeleton = np.zeros_like(mask)
        temp = mask.copy()
        while True:
            eroded = cv2.erode(temp, None)
            dilated = cv2.dilate(eroded, None)
            diff = cv2.subtract(temp, dilated)
            skeleton = cv2.bitwise_or(skeleton, diff)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break
        return skeleton


# -------------------------- 2. 节点预处理（去重+统一ID） --------------------------
def _prepare_nodes(
    pins: list,
    endpoints: list,
    merge_thresh: float = 3.0
) -> Tuple[List[dict], np.ndarray]:
    """
    合并pin和折线点为统一节点列表，分配唯一ID，近距离自动去重
    优先级：pin > endpoint（重合时保留pin类型）
    
    返回：
        nodes: 节点列表，每个节点含 id/type/x/y/score
        node_coords: (N,2) 坐标数组
    """
    raw_nodes = []
    # 装入引脚
    for p in pins:
        raw_nodes.append({
            "type": "pin",
            "x": float(p.xy[0]),
            "y": float(p.xy[1]),
            "score": float(p.score)
        })
    # 装入折线点/端点
    for e in endpoints:
        raw_nodes.append({
            "type": "endpoint",
            "x": float(e.xy[0]),
            "y": float(e.xy[1]),
            "score": float(e.score)
        })

    if not raw_nodes:
        return [], np.empty((0, 2))

    nodes = []
    used = [False] * len(raw_nodes)

    for i in range(len(raw_nodes)):
        if used[i]:
            continue
        # 聚类距离小于阈值的节点
        cluster = [i]
        used[i] = True
        cx, cy = raw_nodes[i]["x"], raw_nodes[i]["y"]

        for j in range(i + 1, len(raw_nodes)):
            if used[j]:
                continue
            dx = raw_nodes[j]["x"] - cx
            dy = raw_nodes[j]["y"] - cy
            if dx * dx + dy * dy < merge_thresh * merge_thresh:
                cluster.append(j)
                used[j] = True

        # 聚类中心
        avg_x = np.mean([raw_nodes[k]["x"] for k in cluster])
        avg_y = np.mean([raw_nodes[k]["y"] for k in cluster])
        # 类型优先pin
        types = {raw_nodes[k]["type"] for k in cluster}
        node_type = "pin" if "pin" in types else "endpoint"
        max_score = max(raw_nodes[k]["score"] for k in cluster)

        nodes.append({
            "id": len(nodes),  # 全局唯一节点ID
            "type": node_type,
            "x": avg_x,
            "y": avg_y,
            "score": max_score
        })

    node_coords = np.array([[n["x"], n["y"]] for n in nodes], dtype=np.float32)
    return nodes, node_coords


# -------------------------- 3. 节点吸附到导线骨架 --------------------------
def _snap_nodes_to_skeleton(
    nodes: List[dict],
    skeleton: np.ndarray,
    max_dist: int = 10
) -> List[dict]:
    """将检测偏移的节点吸附到最近的骨架像素，保证路径连通"""
    h, w = skeleton.shape
    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        return nodes

    skeleton_pts = np.column_stack((xs, ys)).astype(np.float32)

    for node in nodes:
        px, py = node["x"], node["y"]
        dist_sq = np.sum((skeleton_pts - [px, py]) ** 2, axis=1)
        min_idx = int(np.argmin(dist_sq))
        min_dist = np.sqrt(dist_sq[min_idx])
        if min_dist <= max_dist:
            node["x"] = float(skeleton_pts[min_idx, 0])
            node["y"] = float(skeleton_pts[min_idx, 1])

    return nodes


# -------------------------- 4. 器件障碍物掩码 --------------------------
def _build_obstacle_mask(
    shape: Tuple[int, int],
    comps: list,
    pin_coords: np.ndarray,
    pin_margin: float = 5.0
) -> np.ndarray:
    """
    生成障碍物掩码：器件bbox内部不可通行，引脚位置周围开孔
    True = 障碍物，False = 可通行
    """
    h, w = shape
    obstacle = np.zeros((h, w), dtype=bool)

    # 填充所有器件区域为障碍物
    for comp in comps:
        x1, y1, x2, y2 = map(int, comp.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        obstacle[y1:y2, x1:x2] = True

    # 引脚周围清除障碍物，允许导线连接引脚
    for px, py in pin_coords:
        px, py = int(px), int(py)
        x1 = max(0, px - int(pin_margin))
        y1 = max(0, py - int(pin_margin))
        x2 = min(w, px + int(pin_margin))
        y2 = min(h, py + int(pin_margin))
        obstacle[y1:y2, x1:x2] = False

    return obstacle

# -------------------------- 6. 连通分量分组 -> 生成Net --------------------------
def _build_nets(grid_edges, element_list):
    # ========= 将 grid_edges 映射回真实图像坐标edges =========
    edges = {}
    for j, net_segs in enumerate(grid_edges):
        all_node_idxs = set()
        all_pins = list()
        net_points = {}

        for i, seg in enumerate(net_segs):
            s_typ, s_obj = element_list[seg[0]]
            e_typ, e_obj = element_list[seg[1]]
            all_node_idxs.add(seg[0])
            all_node_idxs.add(seg[1])
            net_points[f"edge_{i+1}"] = [
                {"x": float(s_obj['x']), "y": float(s_obj['y'])},
                {"x": float(e_obj['x']), "y": float(e_obj['y'])}
            ]

        for idx in all_node_idxs:
            typ, obj = element_list[idx]
            if obj['type'] == "pin":
                all_pins.append(obj)

        edges[f'net_{j}'] = {
            "all_pins": all_pins,
            "all_node_idxs": all_node_idxs,
            "edges": net_points,
        }

    return edges


def cluster_coords(coords: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """
    对一维坐标聚类，返回聚类中心，以及每个原始坐标映射到的 cluster 索引
    :param coords: 一维数组 [x0,x1,x2...]
    :param tol: 像素容差，距离小于 tol 归为同一格
    :return: centers (聚类中心数组), idx_map (每个输入 coords 对应的 cluster id)
    """
    if len(coords) == 0:
        return np.array([]), np.array([])
    sort_idx = np.argsort(coords)
    sorted_c = coords[sort_idx]
    centers = []
    idx_map = np.zeros_like(coords, dtype=int)
    current_cluster = [sorted_c[0]]
    for val in sorted_c[1:]:
        if val - np.mean(current_cluster) <= tol:
            current_cluster.append(val)
        else:
            c = np.mean(current_cluster)
            centers.append(c)
            current_cluster = [val]
    # 最后一组
    c = np.mean(current_cluster)
    centers.append(c)
    centers = np.array(centers)
    # 映射每个原始坐标到最近聚类中心 id
    for i, v in enumerate(coords):
        dist = np.abs(centers - v)
        idx_map[i] = np.argmin(dist)
    return centers, idx_map


def build_element_grid(nodes: list, comps: list, pixel_tol=3.0):
    """
    构建二维网格矩阵：node 建图逻辑不变；comp 改为从合并后的 node 向上下左右发射线，在交点处标记 comp
    :param nodes: 节点列表，每个 node {"id":...,"x":float,"y":float,"type":...}
    :param comps: 元件列表，每个 comp 包含 bbox 属性 (x1, y1, x2, y2，x1<x2,y1<y2)
    :param pixel_tol: 坐标聚类像素容差
    :return:
        grid: np.ndarray int，grid[yi,xi] = element_list 下标；空 = -1
        element_list: list，全部 node 在前，comps 在后
        x_centers: np.ndarray，网格每个 xi 对应的真实 x 坐标
        y_centers: np.ndarray，网格每个 yi 对应的真实 y 坐标
    """
    # ========== 1. 构建 element_list（与原逻辑完全一致） ==========
    element_list = []
    for nd in nodes:
        element_list.append(("node", nd))
    comp_start_idx = len(element_list)
    for c in comps:
        element_list.append(("comp", c))

    # 无节点直接返回空
    if len(nodes) == 0:
        return np.full((0, 0), -1), element_list, np.array([]), np.array([])

    # ========== 2. Node 初始聚类（与原逻辑完全一致，得到合并后的 node 坐标） ==========
    node_xs = np.array([nd["x"] for nd in nodes], dtype=float)
    node_ys = np.array([nd["y"] for nd in nodes], dtype=float)
    x_centers_node, x_idx_map = cluster_coords(node_xs, pixel_tol)
    y_centers_node, y_idx_map = cluster_coords(node_ys, pixel_tol)

    # 得到合并后的去重 node 坐标（聚类中心坐标，避免重复发射射线）
    merged_node_set = set()
    for i in range(len(nodes)):
        xi = x_idx_map[i]
        yi = y_idx_map[i]
        gx = x_centers_node[xi]
        gy = y_centers_node[yi]
        merged_node_set.add((gx, gy))
    merged_node_points = list(merged_node_set)

    # ========== 3. 射线法计算 comp 交点（核心修改部分） ==========
    intersections = []  # 格式：(x坐标, y坐标, 对应comp在element_list中的索引)

    # 遍历所有 comp
    for elem_idx, (typ, obj) in enumerate(element_list):
        if typ != "comp":
            continue
        x1, y1, x2, y2 = obj.bbox

        # 从每个合并后的 node 点向四个方向发射线
        for (gx, gy) in merged_node_points:
            # --- 水平向右射线（y=gy，x >= gx）---
            if y1 <= gy <= y2:
                if x1 >= gx:  # 左边界交点
                    intersections.append((x1, gy, elem_idx))
                if x2 >= gx:  # 右边界交点
                    intersections.append((x2, gy, elem_idx))

            # --- 水平向左射线（y=gy，x <= gx）---
            if y1 <= gy <= y2:
                if x1 <= gx:  # 左边界交点
                    intersections.append((x1, gy, elem_idx))
                if x2 <= gx:  # 右边界交点
                    intersections.append((x2, gy, elem_idx))

            # --- 垂直向上射线（x=gx，y >= gy）---
            if x1 <= gx <= x2:
                if y1 >= gy:  # 下边界交点
                    intersections.append((gx, y1, elem_idx))
                if y2 >= gy:  # 上边界交点
                    intersections.append((gx, y2, elem_idx))

            # --- 垂直向下射线（x=gx，y <= gy）---
            if x1 <= gx <= x2:
                if y1 <= gy:  # 下边界交点
                    intersections.append((gx, y1, elem_idx))
                if y2 <= gy:  # 上边界交点
                    intersections.append((gx, y2, elem_idx))

    # ========== 4. 生成最终网格坐标轴（node坐标 + comp交点坐标一起聚类） ==========
    if len(intersections) > 0:
        inter_xs = np.array([p[0] for p in intersections], dtype=float)
        inter_ys = np.array([p[1] for p in intersections], dtype=float)
        all_xs = np.concatenate([node_xs, inter_xs])
        all_ys = np.concatenate([node_ys, inter_ys])
    else:
        all_xs = node_xs
        all_ys = node_ys

    x_centers, _ = cluster_coords(all_xs, pixel_tol)
    y_centers, _ = cluster_coords(all_ys, pixel_tol)
    nx = len(x_centers)
    ny = len(y_centers)

    # 初始化 grid
    grid = np.full((ny, nx), fill_value=-1, dtype=np.int32)

    # ========== 5. 填入 node（与原逻辑完全一致，优先级最高） ==========
    for elem_idx, typ, obj in ((i, t, o) for i, (t, o) in enumerate(element_list) if t == "node"):
        nd = obj
        x = nd["x"]
        y = nd["y"]
        xi = np.argmin(np.abs(x_centers - x))
        yi = np.argmin(np.abs(y_centers - y))
        grid[yi, xi] = elem_idx

    # ========== 6. 填入 comp 交点（不覆盖已有的 node） ==========
    for (x, y, comp_idx) in intersections:
        xi = np.argmin(np.abs(x_centers - x))
        yi = np.argmin(np.abs(y_centers - y))
        if grid[yi, xi] == -1:
            grid[yi, xi] = comp_idx

    return grid, element_list, x_centers, y_centers


def dfs_grid(grid, element_list, wire_mask, x_centers, y_centers):
    # import pdb; pdb.set_trace()
    def get_element(yi, xi):
        idx = grid[yi, xi]
        if idx == -1:
            return None
        typ, obj = element_list[idx]
        return typ, obj, idx

    def is_wire_connect(start_x, start_y, end_x, end_y, thres=0.9):
        start_x = int(round(start_x))
        start_y = int(round(start_y))
        end_x = int(round(end_x))
        end_y = int(round(end_y))

        if end_x < start_x:
            start_x, end_x = end_x, start_x
        if end_y < start_y:
            start_y, end_y = end_y, start_y

        if end_y - start_y > end_x - start_x:
            mid_x = ( end_x + start_x ) // 2
            if (wire_mask[start_y:end_y, mid_x] != 0).sum() > ( end_y - start_y ) * thres:
                return True
        else:
            mid_y = ( end_y + start_y ) // 2
            if (wire_mask[mid_y, start_x:end_x] != 0).sum() > ( end_x - start_x ) * thres:
                return True
            
        return False

    # DFS递归函数
    def dfs(yi, xi, last_idx=None, last_move=None):

        if visited_grid[yi, xi]:
            return
        
        visited_grid[yi, xi] = True

        elem_info = get_element(yi, xi)
        if elem_info is not None:
            typ, obj, elem_idx = elem_info
            # 情况1：碰到comp，修复遗漏的pin
            if typ == "comp":
                new_id = len(element_list)
                fix_pin = {
                    "id": new_id,
                    "type": "pin",
                    "x": x_centers[xi],
                    "y": y_centers[yi],
                    "score": None
                }
                element_list.append(("node", fix_pin))
                segs.append((last_idx, new_id))
            
            # 情况2：碰到node
            elif typ == "node":
                node_used[elem_idx] = True
                if last_idx is not None:
                    segs.append((last_idx, elem_idx))

                # 遇到endpoint：保存当前路径作为grid_edge，然后继续DFS（endpoint是拐点，可以继续向外延伸）
                if obj["type"] == "endpoint" or last_idx is None:
                    # 从endpoint继续向4个方向探索分支
                    for dy, dx in dirs:
                        nyi = yi + dy
                        nxi = xi + dx
                        if 0 <= nyi < ny and 0 <= nxi < nx and not visited_grid[nyi, nxi] and is_wire_connect(x_centers[xi], y_centers[yi], x_centers[nxi], y_centers[nyi]):
                            dfs(nyi, nxi, elem_idx, (dy, dx))

        # 当前是空网格（-1，导线网格），继续沿着之前走向探索
        elif last_move is not None:
            nyi = yi + last_move[0]
            nxi = xi + last_move[1]
            if 0 <= nyi < ny and 0 <= nxi < nx and not visited_grid[nyi, nxi] and is_wire_connect(x_centers[xi], y_centers[yi], x_centers[nxi], y_centers[nyi]):
                dfs(nyi, nxi, last_idx, last_move)


    # 定义 dfs 属性
    ny, nx = grid.shape
    grid_edges = []
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    node_used = [False] * len(element_list)

    # ========= 遍历所有 pin 以及剩余 endpoints，找到所有 net =========
    start_type = ['pin', 'endpoint']
    for time in range(2):
        for yi in range(ny):
            for xi in range(nx):
                elem_info = get_element(yi, xi)
                if elem_info is None:
                    continue
                typ, obj, elem_idx = elem_info
                if typ == "node" and not node_used[elem_idx] and obj["type"] == start_type[time]:
                    segs = []
                    visited_grid = np.zeros_like(grid, dtype=bool)
                    dfs(yi, xi, last_idx=None, last_move=None)
                    grid_edges.append(segs)

    return grid_edges, element_list


# -------------------------- 主入口函数 --------------------------
def extract_wire_nets(
    img: np.ndarray,
    img_path: str | Path,
    pins: list,
    comps: list,
    endpoints: list,
    snap_max_dist: int = 10,
):
    """
    从原图 + YOLO检测结果（器件/引脚/折线点）重建完整导线网络
    
    Args:
        img: BGR格式原图
        pins: PinBox 列表（引脚检测结果）
        comps: DetBox 列表（器件检测结果）
        endpoints: PinBox 列表（导线折线点/端点检测结果）
        merge_thresh: 节点合并距离阈值（像素）
        snap_max_dist: 节点吸附到骨架的最大偏移距离
        pin_obstacle_margin: 引脚周围避开器件障碍物的距离
    
    Returns:
        结构化网络字典，net独立编号，edge独立编号，点带x/y坐标
    """
    # 1. 导线提取与骨架化
    wire_mask = _get_wire_mask(img)
    visualize_mask(img, wire_mask, img_path)
    
    # 2. 节点统一编码 + 去重
    nodes, node_coords = _prepare_nodes(pins, endpoints)
    if not nodes:
        return {}

    # 3. 节点吸附到导线
    nodes = _snap_nodes_to_skeleton(nodes, wire_mask, max_dist=snap_max_dist)

    # 4. 创建稀疏图
    grid, element_list, x_centers, y_centers = build_element_grid(nodes, comps)
    visualize_grid(grid, x_centers, y_centers, element_list, img, img_path)

    # 5. 追踪所有边
    edges, element_list = dfs_grid(grid, element_list, wire_mask, x_centers, y_centers)

    # 4. 器件障碍物掩码（引脚处开孔）
    # pin_coords = node_coords[[i for i, n in enumerate(nodes) if n["type"] == "pin"]]
    # obstacle = _build_obstacle_mask(
    #     (h, w), comps, pin_coords, pin_margin=pin_obstacle_margin
    # )
    # visualize_nodes_obstacle(img, obstacle, nodes, img_path ,title_prefix="after_snap")
    
    # 5. 追踪所有边
    # edges = _trace_edges(wire_mask, nodes, obstacle)    

    # 6. 连通分量分组 -> 输出网络结构
    nets = _build_nets(edges, element_list)

    return nets
