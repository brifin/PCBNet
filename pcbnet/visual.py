from typing import Any

import cv2
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join
from pathlib import Path
import os
import json

def run_vision_visualize(img_path, img, comps, texts, skel, pins):
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

    # cv2.imwrite(join(save_dir, "pred_wire.png"), skel_vis)


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
    with open(f"outputs/gt/{img_path.parent.name}.json", "r") as f:
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


def run_wire_visualize(out: dict[str, Any], img: np.ndarray, img_path: Path):
    h_img, w_img = img.shape[:2]
    save_dir = f"outputs/viz/{img_path.parent.name}"
    os.makedirs(save_dir, exist_ok=True)

    pred_net_viz = np.ones((h_img, w_img, 3), dtype=np.uint8) * 255
    nets = out.get("nets", {})
    np.random.seed(42)

    def contest_pt_to_img(pt: dict[str, float]) -> tuple[int, int]:
        x = pt["x"]
        y_contest = pt["y"]
        y_img = h_img - y_contest
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
