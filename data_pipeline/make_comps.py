import os
import json
from PIL import Image, ImageDraw
# ==================== 配置参数 ====================
data_dir = "data/Dataset200/200_train_cases"
output_root = "outputs/components"
pins_output_path = os.path.join(output_root, "pins.json")
# 处理模式切换：
#   'crop'      — 按bbox裁剪元件小图（原功能，含padding外扩）并绘制pin点
#   'draw_bbox' — 在原图上绘制bbox+pin点，保存完整标注图
PROCESS_MODE = 'crop'
padding = 20  # 裁剪外扩像素值，draw模式下会同时绘制padding边界
pin_radius = 3  # 引脚圆点半径
draw_pins = False # 是否画出pin点
# ==================================================
# 初始化引脚信息字典
all_pins_info = {}
os.makedirs(output_root, exist_ok=True)
# 遍历每个case子目录
for case_name in os.listdir(data_dir):
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
        print(f"跳过 {case_name}: 缺少 json 或 png 文件")
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
        comp_name = comp_info.get("Name", comp_id)
        comp_name = str(case_name) + '_' + comp_name
        comp_type = comp_info.get("type", "unknown")
        if comp_type == "pin":
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
        # 加padding后的坐标（边界裁剪）
        x1 = max(0, int(round(orig_x1 - padding)))
        y1 = max(0, int(round(orig_y1 - padding)))
        x2 = min(img_width, int(round(orig_x2 + padding)))
        y2 = min(img_height, int(round(orig_y2 + padding)))
        if x2 <= x1 or y2 <= y1:
            print(f"跳过 {comp_id}: bbox 尺寸无效")
            continue
        # 创建分类目录
        type_dir = os.path.join(output_root, comp_type)
        os.makedirs(type_dir, exist_ok=True)
        save_path = os.path.join(type_dir, f"{comp_name.replace('/', '_')}.png")
        # ==================== 模式分支处理 ====================
        comp_pins = pins.get(comp_id, {})
        pin_list = []
        # 先解析所有引脚（统一做y翻转）
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
            pin_list.append({
                "pin_key": pin_key,
                "pinname": pin_name,
                "absolute": {"x": px, "y": py},
                "relative": {"x": rel_x, "y": rel_y}
            })

        if PROCESS_MODE == 'crop':
            # 模式1：裁剪元件小图 + 在小图上绘制引脚点
            crop_img = img.crop((x1, y1, x2, y2))
            if draw_pins:
                draw_crop = ImageDraw.Draw(crop_img)
                for p in pin_list:
                    rx, ry = p["relative"]["x"], p["relative"]["y"]
                    # 画圆点
                    draw_crop.ellipse(
                        [rx-pin_radius, ry-pin_radius, rx+pin_radius, ry+pin_radius],
                        fill="red"
                    )
                    # 绘制引脚名字，偏移一点避免盖住圆点
                    draw_crop.text((rx + 4, ry - 4), p["pinname"], fill="red")
            crop_img.save(save_path)

        elif PROCESS_MODE == 'draw_bbox':
            # 模式2：在原图绘制bbox + 引脚点
            draw_img = img.copy()
            draw = ImageDraw.Draw(draw_img)
            # 红色粗框：原始元件bbox
            draw.rectangle(
                [orig_x1, orig_y1, orig_x2, orig_y2],
                outline='red',
                width=3
            )
            # 蓝色细框：padding外扩边界（裁剪区域）
            draw.rectangle(
                [x1, y1, x2, y2],
                outline='blue',
                width=1
            )
            # 绘制所有引脚（原图全局坐标）
            for p in pin_list:
                ax, ay = p["absolute"]["x"], p["absolute"]["y"]
                draw.ellipse(
                    [ax-pin_radius, ay-pin_radius, ax+pin_radius, ay+pin_radius],
                    fill="red"
                )
                draw.text((ax + 4, ay - 4), p["pinname"], fill="red")
            draw_img.save(save_path)

        # 同名元件只保留第一份引脚信息
        if comp_name not in all_pins_info:
            all_pins_info[comp_name] = {
                "type": comp_type,
                "bbox_origin": [x1, y1, x2, y2],
                "pin_count": len(pin_list),
                "pins": pin_list
            }
    print(f"处理完成: {case_name}")
# 保存引脚汇总json
with open(pins_output_path, "w", encoding="utf-8") as f:
    json.dump(all_pins_info, f, indent=2, ensure_ascii=False)
print(f"\n全部处理完成！模式：{PROCESS_MODE}")
print(f"图片保存在: {output_root}/(type)/")
print(f"引脚信息保存在: {pins_output_path}")
print(f"共提取 {len(all_pins_info)} 种元件的引脚信息")
