from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
# ===================== 参数配置 =====================
BASE_H = 150  # 基准高度
HEIGHT_JITTER = (0.85, 1.15)  # 高度抖动
COMPONENT_HEIGHT_RATIO = {
    # 原有
    "amp": 0.8,
    "led": 0.6,
    "m3螺丝": 0.8,
    "mosfet": 0.8,
    "motor": 0.6,
    "net_bidirection": 0.4,
    "net_short": 0.3,
    "opt": 0.9,
    "r": 0.2,
    "switch": 0.2,
    "testpoint": 0.3,
    "transformer": 1.5,
    "v": 0.2,
    "gnd": 0.2,
    "buzzer": 0.7,
    "crystal_2pin": 0.5,
    "amp_3pin": 0.8,
    "l": 0.2,
    "c": 0.2,
    "block": 1.2,
    "ant": 0.2,
    "esd": 0.2,
    "crystal_4pin": 0.5,
    "transfomer": 1.5, # 拼写错误变体
    "jumper": 0.2,
    "amp_5pin": 0.8,
    "battary": 1.0,
    "crystal": 0.5,
    "bjt": 0.8,
    "fuse": 0.2,
    "d": 0.2,
    "bjt_npn": 0.8,
    "circle_header": 0.3,
    "dc": 0.4,
    "bjt_pnp": 0.8,
    "crystal_3pin": 0.5,
}
MAX_COLS = 6  # 网格一行最多几张图
GAP_X = 30    # 图片水平间隔
GAP_Y = 40    # 图片垂直间隔
TEXT_HEIGHT = 25 # 下方文字区域高度
# ===================================================
def compose_base_comps_grid():
    root_dir = Path("data/base_comps")
    output_path = Path("data/base_comps/base_comps_grid_scaled.png")
    # 1. 查找所有png，过滤掉以 _vis.png 结尾的文件
    all_pngs = list(root_dir.rglob("*.png"))
    png_paths = []
    for p in all_pngs:
        if not p.name.endswith("_vis.png"):
            png_paths.append(p)
    if not png_paths:
        print("❌ base_comps 目录下没有找到符合条件的png图片")
        return
    # 2. 收集图片信息：原图、type目录名、缩放后尺寸
    img_info_list = []
    print("===== 读取到的元件列表 =====")
    for p in png_paths:
        # type子目录名称（base_comps/type/xxx.png）
        type_name = p.parent.name
        if type_name not in COMPONENT_HEIGHT_RATIO:
            print(f"⚠️ [{type_name}] 不在 COMPONENT_HEIGHT_RATIO 字典中，跳过该文件: {p}")
            continue
        im = Image.open(p).convert("RGBA")
        orig_w, orig_h = im.size
        # 计算目标高度
        ratio = COMPONENT_HEIGHT_RATIO[type_name]
        jitter = random.uniform(*HEIGHT_JITTER)
        target_h = BASE_H * ratio * jitter
        # 等比例缩放宽度
        scale = target_h / orig_h
        target_w = orig_w * scale
        # 缩放图片（Lanczos高质量缩放）
        new_im = im.resize((round(target_w), round(target_h)), Image.Resampling.LANCZOS)
        img_info_list.append((new_im, type_name))
        print(f"✅ type:{type_name:15s} file:{p.name:20s} orig({orig_w:4d},{orig_h:4d}) -> scaled({round(target_w):4d},{round(target_h):4d})")
    if not img_info_list:
        print("❌ 没有有效图片")
        return

    # 尝试加载字体，失败则用默认
    try:
        font = ImageFont.truetype("simhei.ttf", 35)
    except:
        font = ImageFont.load_default(20)

    # 3. 网格布局计算（增加间隙+文字高度）
    total_imgs = len(img_info_list)
    rows = (total_imgs + MAX_COLS - 1) // MAX_COLS
    row_total_h = []
    col_max_w = [0] * MAX_COLS

    for idx, (im, _) in enumerate(img_info_list):
        r = idx // MAX_COLS
        c = idx % MAX_COLS
        w, h = im.size
        if w > col_max_w[c]:
            col_max_w[c] = w

    for r in range(rows):
        row_max_h = 0
        for c in range(MAX_COLS):
            idx = r * MAX_COLS + c
            if idx >= total_imgs:
                break
            im, _ = img_info_list[idx]
            h = im.height
            if h > row_max_h:
                row_max_h = h
        # 一行总高度：图片高度 + 文字高度 + 底部垂直间隙
        row_total_h.append(row_max_h + TEXT_HEIGHT + GAP_Y)

    canvas_w = sum(col_max_w) + GAP_X * (MAX_COLS -1)
    canvas_h = sum(row_total_h)
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # 4. 粘贴图片 + 绘制下方type文字
    y_off = 0
    for r in range(rows):
        x_off = 0
        row_img_max_h = 0
        # 先获取本行最大图片高度
        for c in range(MAX_COLS):
            idx = r * MAX_COLS + c
            if idx >= total_imgs:
                break
            im, _ = img_info_list[idx]
            if im.height > row_img_max_h:
                row_img_max_h = im.height

        for c in range(MAX_COLS):
            idx = r * MAX_COLS + c
            if idx >= total_imgs:
                break
            im, type_name = img_info_list[idx]
            w, h = im.size
            # 垂直居中对齐本行图片
            paste_y = y_off + (row_img_max_h - h)
            canvas.paste(im, (x_off, paste_y), mask=im)
            # 在图片下方绘制type文本，居中
            text_bbox = draw.textbbox((0,0), type_name, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_x = x_off + (w - text_w)/2
            text_y = y_off + row_img_max_h
            draw.text((text_x, text_y), type_name, font=font, fill=(0,0,0,255))

            x_off += col_max_w[c] + GAP_X
        y_off += row_total_h[r]

    canvas.save(output_path)
    print(f"\n🎉 合并完成！输出文件：{output_path}，画布大小 W={canvas.width}, H={canvas.height}")

if __name__ == "__main__":
    compose_base_comps_grid()
