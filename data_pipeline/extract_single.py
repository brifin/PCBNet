import os
os.environ["ORT_DISABLE_CUDA"] = "1"
import json
from PIL import Image, ImageDraw
from rembg import remove
import vtracer
from tqdm import tqdm
from pathlib import Path

# ==================== 单个提取配置 ====================
target_comp_name = "0039_U2"

# ==================== 修复 bbox 相对坐标 ====================
fix_bbox_left_padding     =  0
fix_bbox_right_padding    =  0
fix_bbox_down_padding     =  0
fix_bbox_up_padding       =  0

# ==================== 修复 pin 相对坐标 ====================
fix_pin = {
        "B": [
          0,
          1
        ],
        "E": [
          6,
          0
        ],
        "C": [
          6,
          0
        ]
}

# 是否启动图形学去背景，否为 AI，精度更高速度更慢，优先推荐 True
USE_THRESHOLD_BG = False

# 引脚外扩边距：裁剪PNG时，在最外侧引脚外额外保留的像素，建议 1~5px
CROP_MARGIN = 2
BBOX_GAP = 0

# 可视化开关：True=绘制bbox和pin点并保存vis图，False=不绘制
DRAW_PIN_AND_BBOX = True

# ======================================================


data_dir = "data/Dataset200/200_train_cases"
output_root = "outputs/svg"

def png2svg(img, use_threshold_bg=USE_THRESHOLD_BG, gray_threshold=230):
    """
    :param img: PIL原图(RGB)
    :param name: 元件名称
    :param use_threshold_bg: True=灰度阈值抠白色背景；False=rembg AI抠图
    :param gray_threshold: 灰度阈值，>该值的像素变为透明（白色背景）
    """
    if use_threshold_bg:
        # 转RGBA，增加alpha通道
        img_rgba = img.convert("RGBA")
        # 转灰度图用于判断
        gray = img.convert("L")
        # 【优化：替换慢速双层for循环，使用PIL point矩阵操作】
        mask = gray.point(lambda v: 0 if v > gray_threshold else 255)
        img_rgba.putalpha(mask)
        img_nobg = img_rgba
    else:
        # 原有rembg AI抠除背景
        img_nobg = remove(img)
    img_nobg.save(OUTPUT_TRANSPARENT_PNG)
    # ===== 后面保留你原来注释掉的vtracer矢量化代码不变 =====
    # 2. 将透明PNG矢量化为SVG
    # svg_str = vtracer.convert_image_to_svg_py(
    #     OUTPUT_TRANSPARENT_PNG,
    #     OUTPUT_SVG,
    #     colormode="color",      # color / binary
    #     hierarchical="stacked",
    #     mode="spline",
    #     filter_speckle=4,       # 去除噪点小色块
    #     color_precision=8,
    #     layer_difference=16,
    #     corner_threshold=60,
    #     length_threshold=4.0,
    #     max_iterations=10,
    #     splice_threshold=45,
    #     path_precision=3
    # )


def upsert_comp_json(json_path: str, new_entry: dict):
    """
    顶层key = comp_name，更新/新增元件记录
    :param json_path: 输出json文件路径
    :param new_entry: 单条元件完整dict，必须包含 "comp_name"
    """
    p = Path(json_path)
    comp_name = new_entry["comp_name"]
    # 加载已有数据
    data = {}
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    # 直接覆盖/新增
    data[comp_name] = new_entry
    # 写回
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return


all_pins_info = {}
os.makedirs(output_root, exist_ok=True)
target_case_name = target_comp_name.split("_")[0]

# 遍历每个case子目录
for case_name in tqdm(os.listdir(data_dir)):
    if case_name != target_case_name:
        continue
    case_dir = os.path.join(data_dir, case_name)
    if not os.path.isdir(case_dir):
        continue
    # 自动识别json和png文件
    json_file = None
    img_file = None
    for f in os.listdir(case_dir):
        if f.lower().endswith(".json"):
            json_file = os.path.join(case_dir, f)
        elif f.lower().endswith(".png"):
            img_file = os.path.join(case_dir, f)
    if not json_file or not img_file:
        # print(f"跳过 {case_name}: 缺少 json 或 png 文件")
        continue
    # 读取标注数据
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    components = data.get("components", {})
    pins = data.get("pins", {})
    # 打开原图
    img = Image.open(img_file)
    img_width, img_height = img.size
    # 逐个处理元件
    for comp_id, comp_info in components.items():
        comp_name = comp_info.get("Name", comp_id).replace("/", "_")
        comp_name_full = str(case_name) + '_' + comp_name
        comp_type = comp_info.get("type", "unknown")
        if comp_type == "pin":
            continue
        if comp_name_full != target_comp_name:
            continue
        # ========== 1. 读取元件本体bbox，转换为图像坐标系 ==========
        bbox = comp_info.get("bbox", None)
        if not bbox or len(bbox) != 4:
            print(f"跳过 {comp_id}: bbox 格式错误")
            continue
        orig_x1, orig_y1_world, orig_x2, orig_y2_world = bbox
        # y轴翻转：世界坐标（向上）转图像坐标（向下）
        comp_img_y1 = img_height - orig_y1_world
        comp_img_y2 = img_height - orig_y2_world
        # 保证 y1 < y2
        comp_img_y1, comp_img_y2 = min(comp_img_y1, comp_img_y2), max(comp_img_y1, comp_img_y2)
        comp_img_x1, comp_img_x2 = orig_x1, orig_x2
        comp_img_x1 = max(0, int(round(comp_img_x1 - fix_bbox_left_padding)))
        comp_img_y1 = max(0, int(round(comp_img_y1 - fix_bbox_up_padding)))
        comp_img_x2 = min(img_width, int(round(comp_img_x2 + fix_bbox_right_padding)))
        comp_img_y2 = min(img_height, int(round(comp_img_y2 + fix_bbox_down_padding)))
        # ========== 2. 提前解析所有引脚的绝对图像坐标 ==========
        comp_pins = pins.get(comp_id, {})
        pin_img_coords = []  # 存储所有引脚的 (px, py) 图像坐标
        save_pins = {}
        save_fix_pins = {}
        if comp_pins:
            for pin_key, pin_info in comp_pins.items():
                pin_name = pin_info.get("pinname", pin_key)
                point = pin_info.get("point", {})
                px_world = point.get("x", 0.0)
                py_world = point.get("y", 0.0)
                if pin_name in fix_pin:
                    px_world += fix_pin[pin_name][0]
                    py_world -= fix_pin[pin_name][1]
                    save_fix_pins[pin_name] = fix_pin[pin_name]
                # 引脚y坐标翻转到图像坐标系
                px = px_world
                py = img_height - py_world
                pin_img_coords.append((px, py))
                # 先暂存绝对坐标，相对坐标后面计算
                save_pins[pin_key] = {
                    "pinname": pin_name,
                    "absolute": {"x": px, "y": py},
                    "relative": {}
                }
        # ========== 3. 基于引脚方向计算裁剪框（CROP_MARGIN：图片留白） ==========
        # 3.1 判断每个pin离原始元件本体bbox哪个边界最近
        left_pin_xs = []   # 归属【元件左边界】的引脚x
        right_pin_xs = []  # 归属【元件右边界】的引脚x
        top_pin_ys = []    # 归属【元件上边界】的引脚y
        bottom_pin_ys = [] # 归属【元件下边界】的引脚y
        for px, py in pin_img_coords:
            # 引脚距离【原始元件本体bbox】四边距离
            dist_left = px - comp_img_x1
            dist_right = comp_img_x2 - px
            dist_top = py - comp_img_y1
            dist_bottom = comp_img_y2 - py
            min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
            if min_dist == dist_left:
                left_pin_xs.append(px)
            elif min_dist == dist_right:
                right_pin_xs.append(px)
            elif min_dist == dist_top:
                top_pin_ys.append(py)
            else:
                bottom_pin_ys.append(py)
        # 3.2 计算图片裁剪边界：有引脚则用外侧引脚+CROP_MARGIN；无引脚直接用本体bbox
        if comp_pins:
            if left_pin_xs:
                crop_x1 = min(left_pin_xs) - CROP_MARGIN
            else:
                crop_x1 = comp_img_x1
            if right_pin_xs:
                crop_x2 = max(right_pin_xs) + CROP_MARGIN
            else:
                crop_x2 = comp_img_x2
            if top_pin_ys:
                crop_y1 = min(top_pin_ys) - CROP_MARGIN
            else:
                crop_y1 = comp_img_y1
            if bottom_pin_ys:
                crop_y2 = max(bottom_pin_ys) + CROP_MARGIN
            else:
                crop_y2 = comp_img_y2
        else:
            # 无引脚：裁剪框直接使用本体bbox+fix padding
            crop_x1 = comp_img_x1
            crop_y1 = comp_img_y1
            crop_x2 = comp_img_x2
            crop_y2 = comp_img_y2
        # 3.3 裁剪框取整 + 钳制原图边界
        crop_x1 = max(0, int(round(crop_x1)))
        crop_y1 = max(0, int(round(crop_y1)))
        crop_x2 = min(img_width, int(round(crop_x2)))
        crop_y2 = min(img_height, int(round(crop_y2)))
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            print(f"跳过 {comp_id}: 裁剪尺寸无效")
            continue
        # ========== 4. 构造comp_bbox ==========
        if comp_pins:
            # 有引脚：原有逻辑，comp_bbox向内缩进，和引脚保留BBOX_GAP间隙
            if left_pin_xs:
                nearest_left_pin_x = max(left_pin_xs)
                abs_bbox_x1 = nearest_left_pin_x + BBOX_GAP
            else:
                abs_bbox_x1 = comp_img_x1
            if right_pin_xs:
                nearest_right_pin_x = min(right_pin_xs)
                abs_bbox_x2 = nearest_right_pin_x - BBOX_GAP
            else:
                abs_bbox_x2 = comp_img_x2
            if top_pin_ys:
                nearest_top_pin_y = max(top_pin_ys)
                abs_bbox_y1 = nearest_top_pin_y + BBOX_GAP
            else:
                abs_bbox_y1 = comp_img_y1
            if bottom_pin_ys:
                nearest_bottom_pin_y = min(bottom_pin_ys)
                abs_bbox_y2 = nearest_bottom_pin_y - BBOX_GAP
            else:
                abs_bbox_y2 = comp_img_y2
            # 安全校验：防止comp_bbox反向（引脚布局异常导致边界交叉）
            if abs_bbox_x1 >= abs_bbox_x2 or abs_bbox_y1 >= abs_bbox_y2:
                print(f"警告：{comp_id} comp_bbox边界反向，使用原始元件本体bbox兜底")
                abs_bbox_x1 = comp_img_x1
                abs_bbox_y1 = comp_img_y1
                abs_bbox_x2 = comp_img_x2
                abs_bbox_y2 = comp_img_y2
        else:
            # 无引脚场景：comp_bbox直接等于本体bbox
            abs_bbox_x1 = comp_img_x1
            abs_bbox_y1 = comp_img_y1
            abs_bbox_x2 = comp_img_x2
            abs_bbox_y2 = comp_img_y2
        # 转为相对于裁剪图左上角的相对坐标，写入pins.json
        comp_rel_x1 = abs_bbox_x1 - crop_x1
        comp_rel_y1 = abs_bbox_y1 - crop_y1
        comp_rel_x2 = abs_bbox_x2 - crop_x1
        comp_rel_y2 = abs_bbox_y2 - crop_y1
        # 4.2 引脚相对坐标（相对于裁剪图左上角）
        for pin_key in save_pins:
            abs_x = save_pins[pin_key]["absolute"]["x"]
            abs_y = save_pins[pin_key]["absolute"]["y"]
            save_pins[pin_key]["relative"]["x"] = abs_x - crop_x1
            save_pins[pin_key]["relative"]["y"] = abs_y - crop_y1
        # ========== 5. 构建保存字典（新增comp_bbox字段） ==========
        save_dict = {
            "comp_name": target_comp_name,
            "num_pins": len(comp_pins.keys()),
            "comp_bbox": {
                "x1": comp_rel_x1,
                "y1": comp_rel_y1,
                "x2": comp_rel_x2,
                "y2": comp_rel_y2
            },
            "pins": save_pins,
            "fix": {
                "crop_margin": CROP_MARGIN,
                "bbox_gap": BBOX_GAP,
                "bbox": {
                    "left": fix_bbox_left_padding,
                    "right": fix_bbox_right_padding,
                    "down": fix_bbox_down_padding,
                    "up": fix_bbox_up_padding
                },
                "pin": save_fix_pins
            }
        }
        # ========== 6. 保存引脚JSON ==========
        upsert_comp_json("/data1/chenjh/PCBNet/data/base_comps/pins.json", save_dict)
        # ========== 7. 按新裁剪框截取PNG ==========
        crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        # ========== 8. 保存文件 & SVG处理 ==========
        # 创建分类目录
        type_dir = os.path.join("/data1/chenjh/PCBNet/data/base_comps", comp_type)
        os.makedirs(type_dir, exist_ok=True)
        OUTPUT_TRANSPARENT_PNG = os.path.join(type_dir, f"{target_comp_name}.png")
        OUTPUT_SVG = f"{type_dir}/svg.svg"
        # ====================== 【新增可视化绘制逻辑】 ======================
        if DRAW_PIN_AND_BBOX:
            vis_img = img.crop((crop_x1-10, crop_y1-10, crop_x2+10, crop_y2+10)).copy()
            draw = ImageDraw.Draw(vis_img)
            # 绘制comp_bbox：红色矩形
            bbox_xy = [comp_rel_x1+10, comp_rel_y1+10, comp_rel_x2+10, comp_rel_y2+10]
            draw.rectangle(bbox_xy, outline="red", width=1)
            # 有引脚才绘制引脚点
            pin_radius = 2
            for pin_key, pin_info in save_pins.items():
                rx = pin_info["relative"]["x"]+10
                ry = pin_info["relative"]["y"]+10
                # 画圆：左上角、右下角
                circle_box = [rx - pin_radius, ry - pin_radius, rx + pin_radius, ry + pin_radius]
                draw.ellipse(circle_box, fill="blue")
                draw.text((rx-8, ry), pin_info["pinname"], fill="red")
            # 保存可视化图片
            vis_path = os.path.join(type_dir, f"{target_comp_name}_vis.png")
            vis_img.save(vis_path)
        # ===================================================================
        svg_file = png2svg(crop_img)
