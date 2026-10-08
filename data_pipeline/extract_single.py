import os
os.environ["ORT_DISABLE_CUDA"] = "1"

import json
from PIL import Image, ImageDraw
from rembg import remove
import vtracer
from tqdm import tqdm
from pathlib import Path

# ==================== 单个提取配置 ====================
target_comp_name = "0206_∅2"
left_padding = 3
right_padding = 3
down_padding = 0
up_padding = 0

data_dir = "data/Dataset200/200_train_cases"
output_root = "outputs/svg"

def png2svg(img, name):
    # 1. AI抠除背景，输出带透明通道PNG
    img_nobg = remove(img)
    img_nobg.save(OUTPUT_TRANSPARENT_PNG)
    # 2. 将透明PNG矢量化为SVG
    # vtracer参数调优：颜色分层、平滑度
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
    # 保存svg
    # with open(OUTPUT_SVG, "w", encoding="utf-8") as f:
    #     f.write(svg_str)


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

# 遍历每个case子目录
for case_name in tqdm(os.listdir(data_dir)):
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
        bbox = comp_info.get("bbox", None)
        if not bbox or len(bbox) != 4:
            print(f"跳过 {comp_id}: bbox 格式错误")
            continue
        orig_x1, orig_y1_world, orig_x2, orig_y2_world = bbox
        # 上下翻转：y = H - y_world 工程坐标转图像坐标
        orig_y1_img = img_height - orig_y1_world
        orig_y2_img = img_height - orig_y2_world
        # 交换y，保证 y1 < y2 
        orig_y1 = min(orig_y1_img, orig_y2_img)
        orig_y2 = max(orig_y1_img, orig_y2_img)
        orig_x1, orig_x2 = orig_x1, orig_x2

        x1 = max(0, int(round(orig_x1 - left_padding)))
        y1 = max(0, int(round(orig_y1 - up_padding)))
        x2 = min(img_width, int(round(orig_x2 + right_padding)))
        y2 = min(img_height, int(round(orig_y2 + down_padding)))

        if x2 <= x1 or y2 <= y1:
            print(f"跳过 {comp_id}: bbox 尺寸无效")
            continue
        # 创建分类目录
        # type_dir = os.path.join(output_root, comp_type)
        type_dir = os.path.join("/data1/chenjh/PCBNet/data/base_comps", comp_type)
        os.makedirs(type_dir, exist_ok=True)

        save_path = os.path.join(type_dir, f"{target_comp_name}.png")
        OUTPUT_TRANSPARENT_PNG = os.path.join(type_dir, f"{target_comp_name}.png")
        OUTPUT_SVG = f"{type_dir}/svg.svg"
        pins_output_path = os.path.join(type_dir, "pin.json")
        # pins_output_path = os.path.join(type_dir, f"{target_comp_name}_pins.json")

        # ==================== 模式分支处理 ====================
        comp_pins = pins.get(comp_id, {})        # 先解析所有引脚（统一做y翻转）
        save_pins = {}
        for pin_key, pin_info in comp_pins.items():
            pin_name = pin_info.get("pinname", pin_key)
            point = pin_info.get("point", {})
            px_world = point.get("x", 0.0)
            py_world = point.get("y", 0.0)
            # 引脚y坐标翻转到图像坐标系
            px = px_world
            py = img_height - py_world
            # 相对坐标（相对于裁剪图左上角 x1,y1）
            rel_x = px - x1
            rel_y = py - y1
            save_pins[pin_key] = {
                # "pin_key": pin_key,
                "pinname": pin_name,
                "absolute": {"x": px, "y": py},
                "relative": {"x": rel_x, "y": rel_y}
            }

        save_dict = {
            "comp_name": target_comp_name,
            "num_pins": len(comp_pins.keys()),
            "pins": save_pins,
            "padding": {
                "left": left_padding,
                "down": down_padding,
                "right": right_padding,
                "up": up_padding
            }
        }

        # with open(pins_output_path, "w") as f:
        #     json.dump(save_dict, f, ensure_ascii=False)

        upsert_comp_json("/data1/chenjh/PCBNet/data/base_comps/pins.json", save_dict)

        crop_img = img.crop((x1, y1, x2, y2))
        # crop_img.save(save_path)
        svg_file = png2svg(crop_img, comp_name_full)
