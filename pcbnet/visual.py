from typing import Any

import cv2
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join
from pathlib import Path
import os
import json

def run_vision_visualize(img_path, img, comps, texts, skel, pins, endpoints):
    img_path = Path(img_path)
    save_dir = f"outputs/viz/{img_path.parent.name}"
    os.makedirs(save_dir, exist_ok=True)
    img_bgr = img.copy()

    # ========== 图1：组件框(标 type) + 文本框(标 content) + pin 打点 ==========
    from .gt_adapter import load_category_map

    cmap = load_category_map()
    vis_comp = img_bgr.copy()
    for comp in comps:
        x1, y1, x2, y2 = map(int, comp.bbox)
        cv2.rectangle(vis_comp, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(vis_comp, cmap.get(comp.label, comp.label), (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
    for t in texts:
        x1, y1, x2, y2 = map(int, t.bbox)
        cv2.rectangle(vis_comp, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(vis_comp, f"{t.content}", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    for pin in pins:
        x, y = pin.xy
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(vis_comp, (cx, cy), radius=2, color=(255, 0, 255), thickness=-1)

    for ep in endpoints:
        x, y = ep.xy
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(vis_comp, (cx, cy), radius=2, color=(255, 255, 0), thickness=-1)

    vis_comp_rgb = cv2.cvtColor(vis_comp, cv2.COLOR_BGR2RGB)
    fig1 = plt.figure(figsize=(12, 8))
    plt.imshow(vis_comp_rgb)
    plt.axis("off")
    plt.title("Pred Components")
    plt.tight_layout()
    plt.savefig(join(save_dir, "pred_components.png"), bbox_inches="tight", dpi=150)
    plt.close(fig1)

    # ========== 图3：wire骨架二值图 ==========
    # skel_vis = (skel > 0).astype(np.uint8) * 255

    # kernel = np.ones((3, 3), dtype=np.uint8)
    # skel_vis = cv2.dilate(skel_vis, kernel, iterations=1)

    # cv2.imwrite(join(save_dir, "pred_unet.png"), skel_vis)


def flip_gt_y(gt: dict, h_img: int) -> dict:
    gt = gt.copy()

    comps = gt.get("components", {})
    for cid, cinfo in comps.items():
        bbox = cinfo.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cinfo["bbox"] = [
                x1,
                h_img - y2,
                x2,
                h_img - y1
            ]

    pins = gt.get("pins", {})
    for cid, pin_dict in pins.items():
        for pin_key, pin_info in pin_dict.items():
            point = pin_info.get("point")
            if point and "y" in point:
                point["y"] = h_img - point["y"]

    nets = gt.get("nets", {})
    for net_id, net_info in nets.items():
        edges = net_info.get("edges", {})
        for eid, pts in edges.items():
            for pt in pts:
                if "y" in pt:
                    pt["y"] = h_img - pt["y"]

    return gt


def run_gt_visualize(img_path, img):
    img_path = Path(img_path)
    gt_path = Path(f"outputs/gt/{img_path.parent.name}.json")
    if not gt_path.exists():
        # 没有 GT 时跳过，避免整条可视化链因一个缺失文件中断
        print(f"[viz] skip GT visualize: {gt_path} not found")
        return
    with open(gt_path, "r") as f:
        gt = json.load(f)

    save_dir = f"outputs/viz/{img_path.parent.name}"
    os.makedirs(save_dir, exist_ok=True)

    h_img, w_img = img.shape[:2]

    # 关键：图片不动，只转换 GT 坐标
    gt = flip_gt_y(gt, h_img)

    img_bgr = img.copy()

    # ====================== 图1：GT components（组件框标 type + pin 打点）======================
    vis_gt_comp = img_bgr.copy()
    comps_gt = gt.get("components", {})

    for cid, cinfo in comps_gt.items():
        bbox = cinfo["bbox"]
        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(vis_gt_comp, (x1, y1), (x2, y2), (255, 0, 0), 1)

        txt_type = cinfo.get("type", "")
        cv2.putText(vis_gt_comp, txt_type, (x1, max(y1 - 2, 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        txt_name = cinfo.get("Name") if cinfo.get("Name") else " "
        txt_value = cinfo.get("value") if cinfo.get("value") else " " 
        txt_nv = f"{txt_name} {txt_value}"
        cv2.putText(vis_gt_comp, txt_nv, (x1, max(y1 - 12, 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    pins_gt = gt.get("pins", {})
    for cid, pin_dict in pins_gt.items():
        for pin_key, pin_info in pin_dict.items():
            pt = pin_info.get("point")
            if pt is not None:
                px = int(round(pt["x"]))
                py = int(round(pt["y"]))
                cv2.circle(vis_gt_comp, (px, py), radius=2, color=(255, 0, 255), thickness=-1)

    vis_gt_rgb = cv2.cvtColor(vis_gt_comp, cv2.COLOR_BGR2RGB)

    fig1 = plt.figure(figsize=(12, 8))
    plt.imshow(vis_gt_rgb)
    plt.axis("off")
    plt.title("GT Components")
    plt.tight_layout()
    plt.savefig(join(save_dir, "gt_components.png"), bbox_inches="tight", dpi=150)
    plt.close(fig1)

    # ====================== 图2：GT nets（连线 + 点）======================
    gt_net_viz = np.ones((h_img, w_img, 3), dtype=np.uint8) * 255
    nets = gt.get("nets", {})
    np.random.seed(42)  # 固定随机种子，每次相同net颜色不变
    for net_id, net_info in nets.items():
        # 为当前net随机生成BGR颜色，避开纯白
        color = np.random.randint(0, 200, size=3, dtype=int).tolist()
        edges = net_info.get("edges", {})
        for eid, pts in edges.items():
            # 画连线
            for i in range(len(pts) - 1):
                p0 = pts[i]
                p1 = pts[i + 1]
                x0, y0 = int(p0["x"]), int(p0["y"])
                x1, y1 = int(p1["x"]), int(p1["y"])
                cv2.line(gt_net_viz, (x0, y0), (x1, y1), color, thickness=1)
            # 遍历这条edge上所有点，绘制圆点标记
            for pt in pts:
                x, y = int(pt["x"]), int(pt["y"])
                cv2.circle(gt_net_viz, (x, y), radius=2, color=color, thickness=-1)

    # 保存可视化图
    cv2.imwrite(join(save_dir, "gt_wire.png"), gt_net_viz)


def visualize_mask(img: np.ndarray, mask: np.ndarray, img_path: str | Path):
    """
    可视化原图 + mask二值图 + mask叠加在原图上的效果
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.figure(figsize=(6, 6))
    h, w = mask.shape
    plt.figure(figsize=(w / 100, h / 100), dpi=100)


    # subplot1 原图
    # plt.subplot(1, 3, 1)
    # plt.title("原图")
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.axis("off")

    # subplot2 mask 二值图
    # plt.subplot(1, 3, 2)
    # plt.title("Wire Mask (白色=导线前景)")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    # subplot3 mask叠加到原图（绿色高亮mask区域）
    # overlay = img.copy()
    # green = np.zeros_like(img)
    # green[:, :, 1] = mask
    # overlay = cv2.addWeighted(overlay, 0.7, green, 0.4, 0)
    # plt.subplot(1, 3, 3)
    # plt.title("Mask Overlay 叠加图")
    # plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    # plt.axis("off")

    plt.tight_layout()

    save_dir = f"outputs/viz/{Path(img_path).parent.name}"
    plt.savefig(join(save_dir, "mask.png"))
    plt.close()
    

def visualize_skeleton(mask: np.ndarray, skeleton: np.ndarray, img=None, pins=None, endpoints=None):
    """
    mask, skeleton可视化
    :param mask: 二值导线mask (255=前景)
    :param skeleton: Zhang-Suen细化后的单像素骨架
    :param img: 原图BGR(可选，用来叠加对比)
    :param pins: pin点列表（可选）
    :param endpoints: 折线端点列表（可选）
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig = plt.figure(figsize=(18, 6))

    # 2. 单像素骨架
    plt.subplot(1, 2, 1)
    plt.title("Skeleton (Zhang-Suen)")
    plt.imshow(skeleton, cmap="gray")
    plt.axis("off")

    # 3. 骨架叠加原图（红色线条为骨架）
    plt.subplot(1, 2, 2)
    if img is not None:
        overlay = img.copy()
        red = np.zeros_like(img)
        red[:, :, 2] = skeleton  # 红色通道绘制骨架
        overlay = cv2.addWeighted(overlay, 0.7, red, 0.9, 0)
        # 画pin(蓝色圆点)、endpoint(绿色圆点)
        if pins is not None:
            for p in pins:
                x, y = int(p.xy[0]), int(p.xy[1])
                cv2.circle(overlay, (x, y), 4, (255,0,0), -1)
        if endpoints is not None:
            for e in endpoints:
                x, y = int(e.xy[0]), int(p.xy[1])
                cv2.circle(overlay, (x, y), 4, (0,255,0), -1)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(skeleton, cmap="gray")
    plt.title("Skeleton Overlay on Image")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("test.jpg")
    plt.close()


def visualize_nodes_obstacle(img: np.ndarray, obstacle: np.ndarray, nodes: list, img_path: str | Path, title_prefix=""):
    """
    可视化：原图 | obstacle掩码(白色=障碍物) | obstacle叠加原图 + nodes标记
    nodes: _prepare_nodes输出的节点列表，每个节点包含 id, type, x, y
    obstacle: bool数组，True=障碍物区域
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    h, w = img.shape[:2]

    # 转uint8可视化掩码
    obstacle_uint8 = (obstacle.astype(np.uint8)) * 255

    # 叠加图：在原图上画障碍物半透明 + nodes
    overlay_img = img.copy()
    # 障碍物区域涂上浅蓝色
    blue_mask = np.zeros_like(img)
    blue_mask[obstacle, 0] = 255
    overlay_img = cv2.addWeighted(overlay_img, 0.8, blue_mask, 0.3, 0)

    # 绘制节点
    for nd in nodes:
        x, y = int(round(nd["x"])), int(round(nd["y"]))
        if not (0 <= x < w and 0 <= y < h):
            continue
        if nd["type"] == "pin":
            # pin: 红色实心圆
            cv2.circle(overlay_img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(overlay_img, f"{nd['id']}", (x+6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),1)
        elif nd["type"] == "endpoint":
            # endpoint: 绿色实心圆
            cv2.circle(overlay_img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(overlay_img, f"{nd['id']}", (x+6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)

    fig = plt.figure(figsize=(12, 6))

    # subplot2 obstacle掩码
    plt.subplot(1,2,1)
    plt.title("Obstacle Mask (白色=器件障碍物)")
    plt.imshow(obstacle_uint8, cmap="gray")
    plt.axis("off")

    # subplot3 叠加图（障碍物+所有nodes，附带node id文字）
    plt.subplot(1,2,2)
    plt.title("Obstacle + Nodes (pin红 / endpoint绿)")
    plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    save_dir = f"outputs/viz/{Path(img_path).parent.name}"
    plt.savefig(join(save_dir, "pre_edge.png"))
    plt.close()



def visualize_grid(grid, x_centers, y_centers, element_list, img, img_path: str | Path):
    h,w = img.shape[:2]
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(14,10), dpi=150)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0.6)
    ny, nx = grid.shape

    # ========= 绘制网格横竖线（网格单元格边界） =========
    # 计算单元格中间分隔线（相邻两个中心的中点）
    x_edges = []
    for i in range(nx-1):
        mid = (x_centers[i] + x_centers[i+1]) / 2.0
        x_edges.append(mid)
    y_edges = []
    for i in range(ny-1):
        mid = (y_centers[i] + y_centers[i+1]) / 2.0
        y_edges.append(mid)
    # 竖线
    for x in x_edges:
        ax.axvline(x=x, color="#AAAAAA", lw=3, linestyle="--")
    # 横线
    for y in y_edges:
        ax.axhline(y=y, color="#AAAAAA", lw=3, linestyle="--")

    # 绘制器件包围盒（浅蓝色虚线）
    for idx, (typ, obj) in enumerate(element_list):
        if typ == "comp":
            x1, y1, x2, y2 = obj.bbox
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, linestyle="--", color="skyblue", lw=2)
            ax.add_patch(rect)

    # 绘制网格中心点 + index + 坐标gx,gy
    for yi in range(ny):
        gy = y_centers[yi]
        for xi in range(nx):
            gx = x_centers[xi]
            idx = grid[yi,xi]
            if idx == -1:
                continue
            typ, obj = element_list[idx]
            if typ == "node":
                ax.plot(gx,gy, 'ro', markersize=6)
                # label_text = f"{idx}|({gx:.1f},{gy:.1f})"
                label_text = f"{idx}"
                ax.text(gx+4, gy, label_text, color="red", fontsize=10)
            else:
                ax.plot(gx,gy, 'bs', markersize=4)
                # label_text = f"{idx}|({gx:.1f},{gy:.1f})"
                label_text = f"{idx}"
                ax.text(gx+4, gy, label_text, color="blue", fontsize=10)

    ax.set_title("Grid Visualization: grid lines + center coordinates")
    # ax.axis("off")
    plt.tight_layout()

    save_dir = Path("outputs/viz") / Path(img_path).parent.name
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "grid.png"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def run_wire_visualize(nets: dict[str, Any], img: np.ndarray, img_path: Path):
    h_img, w_img = img.shape[:2]
    save_dir = f"outputs/viz/{img_path.parent.name}"
    os.makedirs(save_dir, exist_ok=True)

    pred_net_viz = np.ones((h_img, w_img, 3), dtype=np.uint8) * 255
    np.random.seed(42)

    def contest_pt_to_img(pt: dict[str, float]) -> tuple[int, int]:
        x = pt["x"]
        y_img = pt["y"]
        # y_img = h_img - y_img
        return int(round(x)), int(round(y_img))

    for net_id, net_info in nets.items():
        color = np.random.randint(0, 200, size=3, dtype=int).tolist()
        edges = net_info.get("edges", {})
        for eid, seg in edges.items():
            p0_dict, p1_dict = seg[0], seg[1]
            x0, y0 = contest_pt_to_img(p0_dict)
            x1, y1 = contest_pt_to_img(p1_dict)
            cv2.line(pred_net_viz, (x0, y0), (x1, y1), color, thickness=1)
            cv2.circle(pred_net_viz, (x0, y0), radius=2, color=color, thickness=-1)
            cv2.circle(pred_net_viz, (x1, y1), radius=2, color=color, thickness=-1)
        # if edges:
        #     first_seg = next(iter(edges.values()))
        #     fx, fy = contest_pt_to_img(first_seg[0])
        #     cv2.putText(pred_net_viz, net_id, (fx+3, fy-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    cv2.imwrite(join(save_dir, "pred_wire.png"), pred_net_viz)


_NET_COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0),
    (0, 255, 255), (255, 0, 255), (255, 255, 0),
    (0, 128, 255), (128, 0, 255), (255, 128, 0),
    (0, 0, 128), (0, 128, 0), (128, 0, 0),
    (0, 128, 128), (128, 0, 128), (128, 128, 0),
]


def _pick_net_color(net_idx: int) -> tuple:
    if net_idx < len(_NET_COLORS):
        return _NET_COLORS[net_idx]
    np.random.seed(net_idx)
    return tuple(np.random.randint(30, 230, 3).tolist())


def draw_pcb_pin_net(
    img: np.ndarray,
    pins_dict: dict[str, dict[str, Any]],
    nets_dict: dict[str, Any],
    case_name: str,
    output_root: str = "viz",
    line_thickness: int = 2,
    pin_radius: int = 6,
    pin_color: tuple = (0, 0, 255),
) -> str:
    """把 skele_wire 的 pins/nets 画到原图上并保存。

    :param pins_dict: {"comp_0": {"pin_0": {"point": {"x":, "y":}}}, ...}
    :param nets_dict: {"net_0": {"hyperGraph": "...", "edges": {"edge_0": [...]}}, ...}
    :return: 生成图片的完整路径（<output_root>/<case_name>/pin_net_visual.png）
    """
    out_dir = os.path.join(output_root, case_name)
    os.makedirs(out_dir, exist_ok=True)

    viz = img.copy()
    h, w = img.shape[:2]

    def _net_no(k: str) -> int:
        tail = str(k).rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    for idx, net_name in enumerate(sorted(nets_dict.keys(), key=_net_no)):
        color = _pick_net_color(idx)
        for edge_pts in (nets_dict[net_name].get("edges") or {}).values():
            if len(edge_pts) < 2:
                continue
            pts = [
                (int(np.clip(p["x"], 0, w - 1)), int(np.clip(p["y"], 0, h - 1)))
                for p in edge_pts
            ]
            cv2.polylines(viz, [np.array(pts, dtype=np.int32).reshape((-1, 1, 2))],
                          isClosed=False, color=color, thickness=line_thickness)

    for comp_pins in pins_dict.values():
        for pin_info in comp_pins.values():
            x = int(np.clip(pin_info["point"]["x"], 0, w - 1))
            y = int(np.clip(pin_info["point"]["y"], 0, h - 1))
            cv2.circle(viz, (x, y), pin_radius + 1, (0, 0, 0), thickness=-1)
            cv2.circle(viz, (x, y), pin_radius, pin_color, thickness=-1)

    save_path = os.path.join(out_dir, "pin_net_visual.png")
    cv2.imwrite(save_path, viz)
    return save_path
