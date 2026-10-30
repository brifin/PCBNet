import os
from openai import OpenAI
import cv2
import numpy as np
import base64
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set
import re
import ast


API_KEY = ['sk-7e70fc36aa39491993109a8ce0957a36']

client = OpenAI(
    api_key=API_KEY[0],
    base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
)

PIN_NUMBER = {
    "antenna": 1,
    "battary": 2,
    "bjt": 3,
    "bnp": 2,
    "buzzer": 2,
    "capacitor": 2,
    "diode": 2,
    "esd_protection": 2,
    "fuse": 2,
    "ground": 1,
    "inductor": 2,
    "led": 2,
    "net_input": 1,
    "net_output": 1,
    "net_short": 2,
    "resistor": 2,
    "switch": 2,
    "testpoint": 1,
    "power": 1
}


def img2base64(img_path: str) -> str:
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def cvimg_to_base64(img: np.ndarray) -> str:
    """
    cv2 BGR numpy array -> png base64，带 data:image/png;base64, 前缀，直接喂Qwen VL
    """
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("cv2 imencode png failed")
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def query_llm(text, img_base64=None, enable_thinking=False):
    content = [
                {"type": "input_text", "text": text},
            ]

    if img_base64:
        content.append({"type": "input_image", "image_url": img_base64})

    print('等待模型回复中....')
    response = client.responses.create(
        model="qwen3.8-flash",
        input=[
            {
                "role": "user",
                "content": content
            }
        ],
        extra_body={
            "enable_thinking": enable_thinking
        }
    )

    answer_idx = 1 if enable_thinking else 0
    output_dict = {
        # 'summury': response.output[0].summary[0].text,
        'answer': response.output[answer_idx].content[0].text,
        'num_tokens': response.usage.x_details[0]
    }
    return output_dict


def draw_comp_and_pins(
    img: np.ndarray,
    comp,
    pin: List[Dict[str, Any]],
    save_name: str,
    grid_step: int = 50,
    expand_pad: int = 50,
    save_crop: bool = True
) -> np.ndarray:
    """
    先在全图画全局网格、蓝色元件框、红色pin点、绿色外框，
    再按绿色框裁剪返回；裁剪内部额外补画可见刻度数字。
    """
    canvas = img.copy()
    h, w = canvas.shape[:2]

    # ---------------------- 1. 原图全局网格线 ----------------------
    for x in range(0, w, grid_step):
        cv2.line(canvas, (x, 0), (x, h), (180, 180, 180), 1)
    for y in range(0, h, grid_step):
        cv2.line(canvas, (0, y), (w, y), (180, 180, 180), 1)

    # ---------------------- 2. 计算绿色包围框 ----------------------
    cx1, cy1, cx2, cy2 = comp.bbox
    all_x = [cx1, cx2]
    all_y = [cy1, cy2]
    for p in pin:
        all_x.append(p["x"])
        all_y.append(p["y"])

    big_x1 = min(all_x) - expand_pad
    big_y1 = min(all_y) - expand_pad
    big_x2 = max(all_x) + expand_pad
    big_y2 = max(all_y) + expand_pad

    big_x1 = max(0, int(big_x1))
    big_y1 = max(0, int(big_y1))
    big_x2 = min(w, int(big_x2))
    big_y2 = min(h, int(big_y2))

    # cv2.rectangle(canvas, (big_x1, big_y1), (big_x2, big_y2), (0, 200, 0), 2)

    # ---------------------- 3. 蓝色元件框 + label ----------------------
    c_x1, c_y1, c_x2, c_y2 = map(int, comp.bbox)
    cv2.rectangle(canvas, (c_x1, c_y1), (c_x2, c_y2), (255, 0, 0), 2)
    text = f"{comp.label}"
    cv2.putText(
        canvas, text, (c_x1, c_y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1
    )

    # ---------------------- 4. 红色pin点 + 下标 ----------------------
    for idx, p in enumerate(pin):
        px = int(p["x"])
        py = int(p["y"])
        cv2.circle(canvas, (px, py), 5, (0, 0, 255), -1)
        txt = f"{idx}"
        cv2.putText(
            canvas, txt, (px + 6, py - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
        )

    # ---------------------- 5. 在绿色框内部补画刻度数字 ----------------------
    # x轴刻度
    x_start = ((big_x1 + grid_step - 1) // grid_step) * grid_step
    for x in range(x_start, big_x2, grid_step):
        cv2.putText(
            canvas, f"{x}", (x + 2, big_y1 + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1
        )

    # y轴刻度
    y_start = ((big_y1 + grid_step - 1) // grid_step) * grid_step
    for y in range(y_start, big_y2, grid_step):
        cv2.putText(
            canvas, f"{y}", (big_x1 + 2, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1
        )

    # ---------------------- 6. 按绿色框裁剪 ----------------------
    crop_canvas = canvas[big_y1:big_y2, big_x1:big_x2]

    # 保存
    save_path = Path("outputs/" + save_name)
    os.makedirs(save_path.parent, exist_ok=True)
    if save_crop:
        cv2.imwrite(save_path, crop_canvas)

    return crop_canvas


def fix_pin_prompt(pin, comp, current, suppose):
    return f"""
        这是一张被截取的PCB原理图，里面包含着元器件和线。我们对该元器件图进行识别，
        目前引脚的编号与其位置为 {str(pin)}，该器件的信息为 {str(comp)}。理论上该器件的应该有 {suppose} 个引脚，但现在有 {current} 个。
        请你找出缺少（实际引脚数小于理论）、多余（实际引脚数大于理论）的引脚，并将最终答案按照以下格式放在 <answer> 中：
        
        '<answer> [{{'idx':, 'x': , 'y':, 'type': ,'side': , 'fix_type':}}] </answer>'

        注意：
        1.得到的器件预测框为蓝色，框上的文字为识别的标签。
        2.红色的点表示识别的该器件的引脚，点边上的文字表示该引脚在 list 中的下标idx。引脚表示元器件与外界线的连接通道，通常而言，引脚位于颜色交界处或者分叉路交叉点中心。
        3.灰色表示坐标线，线旁边的灰色数字是该线的刻度值
        4.输出格式中:
            (1)type表示引脚类型，'gradient' 表示颜色分界或者交叉点中心；'dead' 表示线断了产生的引脚
            (2)side 表示引脚引出的方向，有四个取值，'top','bottom','left','right'
            (3)fix_type表示你修正的类型，取值'delete'表示删除，'add'表示增加
        5.删除引脚可以通过判断引脚是否在元器件出去的线上来判断
        6.增加引脚的坐标可以通过已有引脚的坐标以及坐标轴线刻度来判断，引脚坐标之间往往具有对称性（x、y坐标相同），且引脚不能出现在元器件框内部
    """


def snap_add_pin_to_wire(
    wire_endpoint: np.ndarray,
    add_pin: dict,
    search_radius: int = 10
) -> tuple[int, int]:
    """
    对add类型引脚，按side方向吸附到wire_endpoint=1的导线骨架像素
    side left/right：固定x，沿y轴上下搜索最近的1
    side top/bottom：固定y，沿x轴左右搜索最近的1
    :param wire_endpoint: H,W 单通道二值图，1=导线骨架，0=背景
    :param add_pin: 待add的pin dict，必须含有 x,y,side
    :param search_radius: 最大搜索半径（像素）
    :return: (snap_x, snap_y) 吸附后的坐标；找不到导线则返回原坐标
    """
    h, w = wire_endpoint.shape
    x0 = int(round(add_pin["x"]))
    y0 = int(round(add_pin["y"]))
    side = add_pin["side"]

    # 边界保护
    def in_bounds(xx, yy):
        return 0 <= xx < w and 0 <= yy < h

    if side in ("left", "right"):
        # 固定 x = x0，沿着y轴上下扫描
        cx = x0
        if not in_bounds(cx, y0):
            return (x0, y0)
        candidates = []
        for dy in range(1, search_radius + 1):
            # 向上
            ny = y0 - dy
            if in_bounds(cx, ny) and wire_endpoint[ny, cx] == 1:
                candidates.append((cx, ny, dy))
            # 向下
            ny = y0 + dy
            if in_bounds(cx, ny) and wire_endpoint[ny, cx] == 1:
                candidates.append((cx, ny, dy))
            if candidates:
                break
        if candidates:
            # 取距离最近
            candidates.sort(key=lambda t: t[2])
            return candidates[0][0], candidates[0][1]
        else:
            return (x0, y0)

    elif side in ("top", "bottom"):
        # 固定 y = y0，沿着x轴左右扫描
        cy = y0
        if not in_bounds(x0, cy):
            return (x0, y0)
        candidates = []
        for dx in range(1, search_radius + 1):
            # 向左
            nx = x0 - dx
            if in_bounds(nx, cy) and wire_endpoint[cy, nx] == 1:
                candidates.append((nx, cy, dx))
            # 向右
            nx = x0 + dx
            if in_bounds(nx, cy) and wire_endpoint[cy, nx] == 1:
                candidates.append((nx, cy, dx))
            if candidates:
                break
        if candidates:
            candidates.sort(key=lambda t: t[2])
            return candidates[0][0], candidates[0][1]
        else:
            return (x0, y0)
    else:
        # side非法，不吸附
        return (x0, y0)


def apply_pin_fix(wire_endpoint: np.ndarray, pin_list: list[dict], llm_text: str) -> list[dict]:
    """
    从llm_text提取内部列表，应用增删改逻辑：
    1. fix：更新同idx的pin
    2. delete：删除对应idx的pin
    3. add：新增pin，自动分配不冲突的idx（取当前最大idx+1）
    pin_list: 原始pin数组，每个元素至少含有 "idx"
    返回新pin列表（不修改原列表）
    """
    # 正则抓取 ...
    match = re.search(r'<answer>(.*?)</answer>', llm_text, re.DOTALL)
    if match:
        fix_items = eval(match.group(1).strip())
    else:
        return pin_list.copy()

    pin_map = {p["idx"]: p.copy() for p in pin_list}

    add_candidates = []
    for item in fix_items:
        ft = item["fix_type"]
        idx = item["idx"]
        if ft == "delete":
            if idx in pin_map:
                del pin_map[idx]
        elif ft == "add":
            item['x'], item['y'] = snap_add_pin_to_wire(wire_endpoint, item)
            add_candidates.append(item)

    # 分配新增的idx：取现有idx最大值+1；空则从0开始
    if pin_map:
        next_idx = max(pin_map.keys()) + 1
    else:
        next_idx = 0
    for it in add_candidates:
        new_p = {
            "idx": next_idx,
            "x": it["x"],
            "y": it["y"],
            "type": it["type"],
            "side": it["side"],
        }
        pin_map[next_idx] = new_p
        next_idx += 1

    # 按idx升序返回列表
    new_pins = sorted(pin_map.values(), key=lambda d: d["idx"])
    return new_pins


def fix_pin_by_llm(wire_endpoint, img, comps, comp_pins, img_name):
    num_total_tokens = 0
    for idx, (comp, p_key) in enumerate(zip(comps, comp_pins.keys())):
        pin = comp_pins[p_key]
        if comp.label in PIN_NUMBER:
            if len(pin) != PIN_NUMBER[comp.label]:
                pic = draw_comp_and_pins(img, comp, pin, f'viz/{img_name}/crops/{idx}.png')
                continue
                img_base64 = cvimg_to_base64(pic)

                for idx, p in enumerate(pin):
                    p['idx'] = idx

                prompt = fix_pin_prompt(pin, comp, len(pin), PIN_NUMBER[comp.label])
                response = query_llm(prompt, img_base64, enable_thinking=True)
                print(response)
                num_total_tokens += response['num_tokens']['total_tokens']
                new_pin = apply_pin_fix(wire_endpoint, pin, response['answer'])
                comp_pins[p_key] = new_pin
                pic = draw_comp_and_pins(img, comp, new_pin, img_name, f'viz/{img_name}/crops/{idx}_llmfix.png')
    return comp_pins