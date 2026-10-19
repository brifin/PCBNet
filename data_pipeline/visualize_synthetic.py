import os
import json
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from os.path import join


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

    # ========== 新增：翻转text的bbox Y坐标 ==========
    text_dict = gt.get("text", {})
    for txt_key, bbox in text_dict.items():
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            text_dict[txt_key] = [
                x1,
                h_img - y2,
                x2,
                h_img - y1
            ]
    gt["text"] = text_dict
    return gt


def run_synthetic_visualize(img_path: str | Path):
    """单个case可视化：在同一张图上绘制 components框、text文本框、pin点、net连线"""
    img_path = Path(img_path)
    json_candidates = list(img_path.parent.glob("*.json"))
    json_path = json_candidates[0]
    case_name = img_path.parent.name  # case_0, case_1 ...
    # 读图片
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] 无法读取图片 {img_path}，跳过")
        return
    # 读取json
    with open(str(json_path), "r", encoding="utf-8") as f:
        data = json.load(f)
    h_img, w_img = img.shape[:2]
    # Y轴翻转（和GT可视化对齐）
    data = flip_gt_y(data, h_img)
    # 画布改成原图拷贝，所有内容画在原图上
    viz_canvas = img.copy()

    # ========== 绘制 components 组件框、文字、pin引脚点 ==========
    comps_gt = data.get("components", {})
    for cid, cinfo in comps_gt.items():
        bbox = cinfo["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        # 蓝色组件框
        cv2.rectangle(viz_canvas, (x1, y1), (x2, y2), (255, 0, 0), 3)
        txt_type = cinfo.get("type", "")
        cv2.putText(viz_canvas, txt_type, (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1)
        txt_name = cinfo.get("Name") if cinfo.get("Name") else " "
        txt_value = cinfo.get("value") if cinfo.get("value") else " "
        txt_nv = f"{txt_name} {txt_value}"
        cv2.putText(viz_canvas, txt_nv, (x1, max(y1 - 12, 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1)

    # ========== 新增：绘制 text 的bbox，绿色，不写文字 ==========
    text_gt = data.get("text", {})
    for txt_key, bbox in text_gt.items():
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            # 绿色框 (BGR: (0,255,0))，线宽2，比组件框细一点方便区分
            cv2.rectangle(viz_canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 绘制pin引脚（紫色圆点）
    pins_gt = data.get("pins", {})
    for cid, pin_dict in pins_gt.items():
        for pin_key, pin_info in pin_dict.items():
            pt = pin_info.get("point")
            if pt is not None:
                px = int(round(pt["x"]))
                py = int(round(pt["y"]))
                cv2.circle(viz_canvas, (px, py), radius=2, color=(255, 0, 255), thickness=-1)

    # ========== 绘制nets连线和顶点（和原有逻辑不变） ==========
    nets = data.get("nets", {})
    np.random.seed(42)
    for net_id, net_info in nets.items():
        color = np.random.randint(0, 200, size=3, dtype=int).tolist()
        edges = net_info.get("edges", {})
        for eid, pts in edges.items():
            # 画折线
            for i in range(len(pts) - 1):
                p0 = pts[i]
                p1 = pts[i + 1]
                x0, y0 = int(p0["x"]), int(p0["y"])
                x1, y1 = int(p1["x"]), int(p1["y"])
                cv2.line(viz_canvas, (x0, y0), (x1, y1), color, thickness=2)
            # 画顶点圆点
            for pt in pts:
                x, y = int(pt["x"]), int(pt["y"])
                cv2.circle(viz_canvas, (x, y), radius=5, color=color, thickness=-1)

    # 保存 outputs/viz_synthetic/case_n.png
    out_dir = Path("outputs/viz_synthetic")
    out_dir.mkdir(exist_ok=True, parents=True)
    out_png_path = out_dir / f"{case_name}.png"
    cv2.imwrite(str(out_png_path), viz_canvas)
    print(f"✅ Saved {case_name} -> {out_png_path}")


if __name__ == "__main__":
    root = Path("data/Synthetic_Data")
    for case_dir in root.iterdir():
        if not case_dir.is_dir():
            continue
        png_list = list(case_dir.glob("*.png"))
        if len(png_list) == 0:
            print(f"[WARN] {case_dir.name} 没有png文件，跳过")
            continue
        # 取该case下第一张png
        img_path = png_list[0]
        run_synthetic_visualize(img_path)
