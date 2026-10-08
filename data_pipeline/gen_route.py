from PIL import Image

def place_png_on_whiteboard(
    png_path: str,
    out_path: str,
    board_width: int,
    board_height: int,
    img_width: int,
    img_height: int,
    paste_x: int = 0,
    paste_y: int = 0
):
    """
    将透明PNG放到白色白板画布上，支持拉伸
    :param png_path: 输入png路径（支持RGBA透明图）
    :param out_path: 输出图片路径
    :param board_width: 白板画布宽度
    :param board_height: 白板画布高度
    :param img_width: 拉伸后的图片宽度
    :param img_height: 拉伸后的图片高度
    :param paste_x: 粘贴左上角x坐标
    :param paste_y: 粘贴左上角y坐标
    """
    # 1. 创建白色白板画布 RGB
    whiteboard = Image.new("RGB", (board_width, board_height), color="white")

    # 2. 读取透明PNG
    png_img = Image.open(png_path).convert("RGBA")

    # 3. 拉伸图片（不等比拉伸）
    png_resized = png_img.resize((img_width, img_height), Image.Resampling.LANCZOS)

    # 4. 粘贴到白板，使用alpha通道作为mask
    whiteboard.paste(png_resized, (paste_x, paste_y), mask=png_resized)

    # 5. 保存
    whiteboard.save(out_path)
    print(f"已保存到 {out_path}")
    return whiteboard


def overlay_rgba_on_whiteboard(
    rgba_img: Image.Image,
    board_width: int,
    board_height: int,
    target_w: int,
    target_h: int,
    paste_x: int = 0,
    paste_y: int = 0
) -> Image.Image:
    # ========= 关键：alpha通道二值化，清除rembg边缘半透明光晕 =========
    r, g, b, a = rgba_img.split()
    a = a.point(lambda x: 255 if x > 128 else 0)
    rgba_img = Image.merge("RGBA", (r, g, b, a))

    # PCB放大必须用 NEAREST，保留硬边缘，禁止LANCZOS平滑糊化
    resized = rgba_img.resize((target_w, target_h), Image.Resampling.NEAREST)

    # 创建白色白板
    board = Image.new("RGB", (board_width, board_height), "white")
    board.paste(resized, (paste_x, paste_y), mask=resized)
    return board

# ============ 使用示例 ============
if __name__ == "__main__":
    place_png_on_whiteboard(
        png_path="/data1/chenjh/PCBNet/outputs/svg/amp/remove_bkg.png",   # 你的rembg输出透明PNG
        out_path="on_whiteboard.png",
        board_width=2000,
        board_height=2000,
        img_width=1500,
        img_height=1500,
        paste_x=100,
        paste_y=100
    )
