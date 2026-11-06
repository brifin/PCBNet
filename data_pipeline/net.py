"""Generate synthetic circuit diagrams and pin/net annotations.

Run from the project directory with the configured ``py3.12`` environment:

    python net.py

The generator reads ``base_comps`` and writes numbered samples to
``synthetic/``. Existing numbered sample directories are left untouched.
"""

from __future__ import annotations

import heapq
import json
import math
import multiprocessing
import os
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm


# ============================ Adjustable settings ============================

# Paths are relative to this script so it works from any current directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE_COMPS_DIR = PROJECT_DIR / "data/base_comps"
PINS_JSON_PATH = BASE_COMPS_DIR / "pins.json"
OUTPUT_DIR = PROJECT_DIR / "data/Synthetic_Data"
WRITE_DEBUG = False  # 生成后校验比例，训练前删除 debug.json

# Number of requested outputs. Sample IDs use 1-based, four-digit folders.
NUM_SAMPLES = 600
# Number of worker processes used to generate the missing samples.
PROCESS_COUNT = 8

# Canvas dimensions are sampled independently for every sample.
BOARD_WIDTH_RANGE = (1500, 2500)
BOARD_HEIGHT_RANGE = (1000, 1500)

# The box's longer image edge occupies this fraction of the matching canvas edge.
BOX_LONG_EDGE_FRACTION = (0.07, 0.10)
BOX_CENTER_JITTER_FRACTION = 0.045

# Non-box sizing: type-specific height ratio times BASE_COMPONENT_HEIGHT.
BASE_COMPONENT_HEIGHT = 60
COMPONENT_HEIGHT_RATIO = {
    # 由真实训练图（Dataset200 0001-0300 + data/Dataset 298）的
    # bbox 高度 / 图像高度中位数反推，贴近真实图纸比例。
    "amp": 0.8,
    "led": 0.74,
    "m3螺丝": 0.8,
    "mosfet": 0.8,
    "motor": 0.6,
    "net_bidirection": 0.4,
    "net_short": 0.3,
    "opt": 0.9,
    "r": 0.42,
    "switch": 0.68,
    "testpoint": 0.3,
    "transformer": 1.5,
    "v": 0.50,
    "gnd": 0.56,
    "buzzer": 0.5,
    "crystal_2pin": 0.6,
    "amp_3pin": 0.8,
    "l": 0.36,
    "c": 0.60,
    "block": 0.4,
    "ant": 0.8,
    "esd": 0.8,
    "crystal_4pin": 0.6,
    "transfomer": 1.5,  # 拼写错误变体
    "jumper": 0.6,
    "amp_5pin": 0.8,
    "battary": 0.5,
    "bjt": 0.8,
    "fuse": 0.3,
    "d": 0.58,
    "bjt_npn": 0.8,
    "circle_header": 1.0,
    "dc": 0.6,
    "bjt_pnp": 0.8,
    "crystal_3pin": 0.6,
    "net_input": 0.48,
    "net_output": 0.42,
    "mosfet_npn": 0.8,
    # "box": 0.8,
}
HEIGHT_JITTER = (1.0, 1.0)
# 非 box 器件统一按 0.80~1.40 的离散倍数缩放（步长 0.02），不做额外宽高变形。
NON_BOX_DEFORMATION_PROBABILITY = 0.0
DEFORMATION_SCALE_RANGE = (1.0, 1.0)
SCALE_STEPS = tuple(round(0.80 + 0.02 * i, 2) for i in range(31))
ROTATION_STEPS = (0, 90, 180, 270)
_SCALE_QUEUE: list[float] = []
_ROTATION_QUEUE: list[int] = []


def next_scale() -> float:
    global _SCALE_QUEUE
    if not _SCALE_QUEUE:
        _SCALE_QUEUE = list(SCALE_STEPS)
        random.shuffle(_SCALE_QUEUE)
    return _SCALE_QUEUE.pop(0)


def next_rotation() -> int:
    global _ROTATION_QUEUE
    if not _ROTATION_QUEUE:
        _ROTATION_QUEUE = list(ROTATION_STEPS)
        random.shuffle(_ROTATION_QUEUE)
    return _ROTATION_QUEUE.pop(0)

# 每个用例随机放置的器件总数（含 box）。
MIN_COMPONENTS_PER_SAMPLE = 35
MAX_COMPONENTS_PER_SAMPLE = 80
MAX_FREE_PLACEMENT_ATTEMPTS = 240

# 噪声：孤立元件之间的横平竖直模拟连线 + 元件框外的大号随机文字。
COMPONENT_NOISE_PROBABILITY = 0.22
COMPONENT_NOISE_SIGMA_RANGE = (4.0, 12.0)
NEARBY_TEXT_PROBABILITY = 0.30
NEARBY_TEXT_FONT_RANGE = (16, 30)
NEARBY_TEXT_GAP_RANGE = (1, 5)
NOISE_WIRE_WIDTH_RANGE = (2, 4)
LED_PAIR_COUNT_RANGE = (3, 5)
GND_PAIR_COUNT_RANGE = (2, 4)
# case11 风格：成对符号只留 2-6 px 间隙（普通器件安全间距是 30 px）
TIGHT_PAIR_GAP_RANGE = (2, 6)

# Non-box orientation distribution: unchanged, 90, 180, and 270 degrees.
ROTATION_ANGLES = (0, 90, 180, 270)
ROTATION_WEIGHTS = (0.70, 0.10, 0.10, 0.10)

# Layout and routing.
COMPONENT_SAFETY_GAP = 30
BOARD_EDGE_MARGIN = 24
CONTENT_MARGIN_FRACTION = 0.10


def content_margins(board_width: int, board_height: int) -> tuple[int, int]:
    return (
        int(round(board_width * CONTENT_MARGIN_FRACTION)),
        int(round(board_height * CONTENT_MARGIN_FRACTION)),
    )
ROUTE_GRID_SPACING = 30
PIN_EXTENSION_LENGTH = 8
PIN_RECTIFICATION_MAX = 10
PIN_RECTIFICATION_PROBE_LENGTH = 7
WIRE_WIDTH = 3
# Minimum centre-line clearance from other wires and component bboxes.
WIRE_GAP = 8
JUNCTION_RADIUS = 5
BEND_PENALTY = 24
MAX_LAYOUT_ATTEMPTS = 20
MAX_COMPONENT_PLACE_ATTEMPTS = 180
MAX_ASTAR_VISITS = 180_000
# 大 box 只选一部分引脚参与布线；其余引脚保留为开路线/噪声连线，
# 这样 72 pin 的大 IC 也能成功放进画布并覆盖所有 box 样例。
MAX_BOX_PINS_USED = 16
MAX_CONSECUTIVE_SAMPLE_FAILURES = 3

# Device actions for a d-operation. Every pin of each placed device joins the
# anchor box pin's net. Restricting pin count keeps generated nets readable.
DEVICES_PER_D_ACTION = (1, 2)
SECOND_DEVICE_PROBABILITY = 0.28
MAX_PINS_PER_PLACED_COMPONENT = 4
DEVICE_OUTWARD_DISTANCE_RANGE = (72, 240)
DEVICE_PERPENDICULAR_JITTER = 150

# Operation-a open pins use either a solid dot or a cross.
OPEN_MARKER_CHOICES = ("dot", "cross")
OPEN_MARKER_RADIUS = 5
OPEN_MARKER_CROSS_HALF_SIZE = 8
OPEN_MARKER_CROSS_WIDTH = 3

# Exactly one or two routed nets per sample are assigned a deliberate,
# electrically unconnected crossing when the layout can support it.
CROSSING_NET_COUNT_RANGE = (0, 1)
CROSSING_CANDIDATE_LIMIT = 16
CROSSING_ENDPOINT_PAIR_LIMIT = 6
CROSSING_ROUTE_ATTEMPT_LIMIT = 12
B_COMPONENT_CANDIDATE_LIMIT = 24
B_VARIANTS_PER_CANDIDATE = 4

# Palette is selected once per sample. The first palette uses the requested
# red component color and green wires; blue components use black wires.
PALETTE_CHOICES = ("red", "blue")
PALETTE_WEIGHTS = (0.50, 0.50)
PALETTE_COMPONENT_COLORS = {
    "red": (0x8D, 0x1F, 0x1E),
    "blue": (0x24, 0x63, 0xB7),
}
PALETTE_WIRE_COLORS = {
    "red": (0x20, 0x89, 0x29),
    "blue": (0x00, 0x00, 0x00),
}

# b-operation candidates are selected from images with exactly one pin.
# 低检测率类别加权，并给每个 case 设最低出现次数（冷门样例多放几个）。
D_TYPE_WEIGHTS = {
    "c": 2.0,
    "led": 7.0,
    "r": 2.0,
    "v": 10.0,
    "gnd": 10.0,
    "battary": 1,
    "d": 2.5,
    "switch": 3.5,
    "circle_header": 1.0,
    "l": 3.0,
    "net_input": 8.0,
    "net_output": 6.0,
}
OTHER_D_TYPE_WEIGHT = 0.4

# 低 mAP50 类别的样例权重：gnd 0003 占 75%，v 的 0009/0011 各 45%。
TYPE_SAMPLE_WEIGHTS = {
    "gnd": {"0003_∅438": 0.85},
    "v": {"0009_∅124": 0.45, "0011_∅611": 0.45, "0001_∅580": 0.10},
}
COLD_MIN_COUNTS = {
    "v": 12,
    "gnd": 14,
    "r": 3,
    "c": 4,
    "l": 3,
    "led": 10,
    "d": 2,
    "switch": 3,
    "circle_header": 1,
    "net_input": 8,
    "net_output": 4,
}

# Set to an integer for reproducible output, or None for fresh randomness.
RANDOM_SEED = None


Point = tuple[int, int]
Rect = tuple[int, int, int, int]
PinId = tuple[str, str]
Edge = tuple[Point, Point]

_WORKER_CANDIDATES: list[dict] | None = None
_WORKER_PROGRESS_QUEUE = None


# ================================ Data loading ================================

def scan_component_pngs(base_dir: Path) -> dict[str, list[Path]]:
    if not base_dir.is_dir():
        raise FileNotFoundError(f"组件目录不存在: {base_dir}")

    files_by_type: dict[str, list[Path]] = {}
    for type_dir in sorted(base_dir.iterdir()):
        if not type_dir.is_dir():
            continue
        files = sorted(
            path for path in type_dir.glob("*.png")
            if path.is_file() and "vis.png" not in path.name
        )
        if files:
            files_by_type[type_dir.name] = files
    return files_by_type


def load_pins_info(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_candidates(files_by_type: dict[str, list[Path]], pins_info: dict):
    candidates: list[dict] = []
    seen: set[str] = set()

    for comp_type, files in files_by_type.items():
        for image_path in files:
            source_name = image_path.stem
            if source_name in seen:
                continue
            info = pins_info.get(source_name)
            valid = bool(
                info and info.get("pins") and info.get("comp_bbox")
                and all(
                    isinstance(pin.get("relative"), dict)
                    and "x" in pin["relative"]
                    and "y" in pin["relative"]
                    for pin in info["pins"].values()
                )
                and all(key in info["comp_bbox"] for key in ("x1", "y1", "x2", "y2"))
            )
            if not valid:
                # 没有 pins.json 的稀有样例（例如 v/0009_∅124）：用 alpha bbox
                # 生成 comp_bbox，并在 bbox 底边中点补一个伪引脚，只用于合成布局。
                if comp_type == "box":
                    continue
                try:
                    with Image.open(image_path) as image:
                        alpha_box = image.convert("RGBA").getchannel("A").getbbox()
                except Exception:
                    alpha_box = None
                if not alpha_box:
                    continue
                x0, y0, x1, y1 = alpha_box
                if x1 <= x0 or y1 <= y0:
                    continue
                info = {
                    "comp_name": source_name,
                    "num_pins": 1,
                    "comp_bbox": {"x1": float(x0), "y1": float(y0),
                                  "x2": float(x1 - 1), "y2": float(y1 - 1)},
                    "pins": {
                        "pin_1": {
                            "pinname": "P_1",
                            "relative": {
                                "x": float((x0 + x1 - 1) / 2),
                                "y": float(y1 - 1),
                            },
                        }
                    },
                }
            seen.add(source_name)
            candidates.append({
                "source_name": source_name,
                "type": comp_type,
                "path": image_path,
                "info": info,
            })
    return candidates


# ============================== Image transforms ==============================

def rotate_coordinate(x: float, y: float, width: int, height: int, angle: int):
    """Transform pixel coordinates exactly as the matching Pillow transpose."""
    if angle == 0:
        return x, y
    if angle == 90:
        return y, width - 1 - x
    if angle == 180:
        return width - 1 - x, height - 1 - y
    if angle == 270:
        return height - 1 - y, x
    raise ValueError(f"不支持的旋转角度: {angle}")


def recolor_alpha(image: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    solid = Image.new("RGBA", rgba.size, (*color, 255))
    solid.putalpha(alpha)
    return solid


_ALNUM_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _random_alnum() -> str:
    return "".join(random.choice(_ALNUM_CHARS) for _ in range(random.randint(1, 4)))


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=max(8, int(size)))
    except TypeError:
        # Pillow < 10.1 没有 size 参数
        return ImageFont.load_default()


def add_component_noise(image: Image.Image) -> Image.Image:
    """在器件 PNG 的非透明区域叠加少量高斯噪声（alpha 不变）。"""
    array = np.asarray(image.convert("RGBA")).astype(np.int16)
    sigma = random.uniform(*COMPONENT_NOISE_SIGMA_RANGE)
    noise = np.random.normal(0.0, sigma, array[:, :, :3].shape)
    array[:, :, :3] = np.clip(array[:, :, :3] + noise, 0, 255)
    return Image.fromarray(array.astype(np.uint8), "RGBA")


def add_component_text(image: Image.Image) -> Image.Image:
    """在器件 PNG 内部随机位置写黑色英文/数字。"""
    text = _random_alnum()
    font = _load_font(max(8, min(18, image.height // 8)))
    draw = ImageDraw.Draw(image)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
    except Exception:
        return image
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    if text_width + 4 >= image.width or text_height + 4 >= image.height:
        return image
    x = random.randint(1, image.width - text_width - 2)
    y = random.randint(1, image.height - text_height - 2)
    draw.text((x, y), text, font=font, fill=(0, 0, 0, 255))
    return image


def transformed_rect(raw: dict, scale_x: float, scale_y: float,
                     width: int, height: int, angle: int) -> Rect:
    corners = [
        (raw["x1"] * scale_x, raw["y1"] * scale_y),
        (raw["x1"] * scale_x, raw["y2"] * scale_y),
        (raw["x2"] * scale_x, raw["y1"] * scale_y),
        (raw["x2"] * scale_x, raw["y2"] * scale_y),
    ]
    points = [rotate_coordinate(x, y, width, height, angle) for x, y in corners]
    return (
        math.floor(min(point[0] for point in points)),
        math.floor(min(point[1] for point in points)),
        math.ceil(max(point[0] for point in points)),
        math.ceil(max(point[1] for point in points)),
    )


def rectify_pin_point(image: Image.Image, point: Point, body: Rect) -> Point:
    """Nudge a pin at most 10 px along one axis to the center of its lead."""
    x, y = point
    direction = direction_from_body(point, body)
    if direction[0]:
        candidates = [(x, y + offset) for offset in range(-PIN_RECTIFICATION_MAX,
                                                           PIN_RECTIFICATION_MAX + 1)]
    else:
        candidates = [(x + offset, y) for offset in range(-PIN_RECTIFICATION_MAX,
                                                           PIN_RECTIFICATION_MAX + 1)]
    alpha = image.getchannel("A")
    pixels = alpha.load()
    width, height = image.size

    def score(candidate: Point):
        cx, cy = candidate
        hit_count = 0
        for distance in range(2, PIN_RECTIFICATION_PROBE_LENGTH + 1):
            sx = cx - direction[0] * distance
            sy = cy - direction[1] * distance
            if not (0 <= sx < width and 0 <= sy < height):
                continue
            # The 3 px cross-section tolerates antialiasing and slight width
            # changes from scaling while still preferring a lead's centerline.
            for perpendicular in (-1, 0, 1):
                px = sx + (perpendicular if direction[0] == 0 else 0)
                py = sy + (perpendicular if direction[0] != 0 else 0)
                if 0 <= px < width and 0 <= py < height and pixels[px, py] > 0:
                    hit_count += 1
        return hit_count

    best = max(candidates, key=lambda candidate: (score(candidate), -abs(candidate[0] - x) - abs(candidate[1] - y)))
    return best


def prepare_component(candidate: dict, palette_color: tuple[int, int, int],
                      *, is_box: bool = False, box_scale: float | None = None,
                      scale_mult: float | None = None, rotation: int | None = None):
    with Image.open(candidate["path"]) as source:
        image = source.convert("RGBA")
    original_width, original_height = image.size
    used_scale = None

    if is_box:
        if box_scale is None:
            raise ValueError("box_scale is required for a box component")
        target_width = max(2, round(original_width * box_scale))
        target_height = max(2, round(original_height * box_scale))
        angle = 0
    else:
        ratio = COMPONENT_HEIGHT_RATIO.get(candidate["type"], 0.7)
        used_scale = scale_mult if scale_mult is not None else next_scale()
        target_height = max(
            2,
            round(BASE_COMPONENT_HEIGHT * ratio * used_scale),
        )
        target_width = max(2, round(target_height * original_width / original_height))
        if random.random() < NON_BOX_DEFORMATION_PROBABILITY:
            target_width = max(
                2,
                round(target_width * random.uniform(*DEFORMATION_SCALE_RANGE)),
            )
        if random.random() < NON_BOX_DEFORMATION_PROBABILITY:
            target_height = max(
                2,
                round(target_height * random.uniform(*DEFORMATION_SCALE_RANGE)),
            )
        angle = rotation if rotation is not None else next_rotation()

    scale_x = target_width / original_width
    scale_y = target_height / original_height
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    if not is_box:
        image = recolor_alpha(image, palette_color)
    # 随机高斯噪声在旋转前加入；文字噪声改为绘制在元件框外。
    if random.random() < COMPONENT_NOISE_PROBABILITY:
        image = add_component_noise(image)

    transpose = {
        90: Image.Transpose.ROTATE_90,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_270,
    }
    if angle:
        image = image.transpose(transpose[angle])

    local_pins: dict[str, dict] = {}
    local_body = transformed_rect(
        candidate["info"]["comp_bbox"],
        scale_x,
        scale_y,
        target_width,
        target_height,
        angle,
    )
    for pin_key, pin in candidate["info"]["pins"].items():
        relative = pin["relative"]
        x = float(relative["x"]) * scale_x
        y = float(relative["y"]) * scale_y
        x, y = rotate_coordinate(x, y, target_width, target_height, angle)
        point = (int(round(x)), int(round(y)))
        point = rectify_pin_point(image, point, local_body)
        local_pins[pin_key] = {
            "point": point,
            "pinname": str(pin.get("pinname", pin_key)),
        }

    width, height = image.size
    return {
        "source_name": candidate["source_name"],
        "type": candidate["type"],
        "scale_mult": used_scale,
        "image": image,
        "rotation": angle,
        "scale_xy": (scale_x, scale_y),
        "local_pins": local_pins,
        "local_body_bbox": local_body,
        "extent_size": (width, height),
    }


# ================================ Layout helpers ===============================

def translate_rect(rect: Rect, x: int, y: int) -> Rect:
    return rect[0] + x, rect[1] + y, rect[2] + x, rect[3] + y


def expand_rect(rect: Rect, amount: int) -> Rect:
    return rect[0] - amount, rect[1] - amount, rect[2] + amount, rect[3] + amount


def rects_overlap(a: Rect, b: Rect) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def materialize_component(prepared: dict, origin: Point, display_name: str):
    x, y = origin
    width, height = prepared["extent_size"]
    return {
        **prepared,
        "name": display_name,
        "value": prepared["source_name"],
        "origin": (x, y),
        "extent": (x, y, x + width - 1, y + height - 1),
        "bbox": translate_rect(prepared["local_body_bbox"], x, y),
        "pins": {
            key: {
                "pinname": value["pinname"],
                "point": (value["point"][0] + x, value["point"][1] + y),
            }
            for key, value in prepared["local_pins"].items()
        },
    }


def component_content_ok(component: dict, board_size: tuple[int, int]) -> bool:
    """器件本身、body bbox 和所有 pin 都必须在画布中心 80% 区域内。"""
    width, height = board_size
    margin_x, margin_y = content_margins(width, height)
    for x0, y0, x1, y1 in (component["extent"], component["bbox"]):
        if x0 < margin_x or y0 < margin_y or x1 > width - margin_x or y1 > height - margin_y:
            return False
    for pin in component["pins"].values():
        x, y = pin["point"]
        if x < margin_x or x > width - margin_x or y < margin_y or y > height - margin_y:
            return False
    return True


_BOX_QUEUE: list[dict] = []


def choose_box_candidate(candidates: list[dict]) -> dict:
    global _BOX_QUEUE
    boxes = [candidate for candidate in candidates if candidate["type"] == "box"]
    if not boxes:
        raise RuntimeError("base_comps/box 中没有带有效引脚信息的元件")
    # 轮转所有 box 样例，保证每个 box PNG 都被用到。
    if not _BOX_QUEUE:
        _BOX_QUEUE = boxes.copy()
        random.shuffle(_BOX_QUEUE)
    return _BOX_QUEUE.pop(0)


def choose_palette():
    name = random.choices(PALETTE_CHOICES, weights=PALETTE_WEIGHTS, k=1)[0]
    return name, PALETTE_COMPONENT_COLORS[name], PALETTE_WIRE_COLORS[name]


def pick_box_component(candidate: dict, board_size: tuple[int, int],
                       palette_color: tuple[int, int, int]):
    board_width, board_height = board_size
    margin_x, margin_y = content_margins(board_width, board_height)
    with Image.open(candidate["path"]) as source:
        original_width, original_height = source.size
    # matching_edge = board_width if original_width >= original_height else board_height
    matching_edge = board_height
    target_long = matching_edge * random.uniform(*BOX_LONG_EDGE_FRACTION)
    box_scale = target_long / max(original_width, original_height) * next_scale()
    prepared = prepare_component(candidate, palette_color, is_box=True, box_scale=box_scale)
    width, height = prepared["extent_size"]
    if width + 2 * margin_x >= board_width or height + 2 * margin_y >= board_height:
        return None
    center_x = board_width / 2 + random.uniform(-BOX_CENTER_JITTER_FRACTION,
                                                BOX_CENTER_JITTER_FRACTION) * board_width
    center_y = board_height / 2 + random.uniform(-BOX_CENTER_JITTER_FRACTION,
                                                  BOX_CENTER_JITTER_FRACTION) * board_height
    x = round(center_x - width / 2)
    y = round(center_y - height / 2)
    x = min(max(margin_x, x), board_width - margin_x - width)
    y = min(max(margin_y, y), board_height - margin_y - height)
    component = materialize_component(prepared, (x, y), "U1")
    return component if component_content_ok(component, board_size) else None


def make_display_name(comp_type: str, counters: defaultdict[str, int]) -> str:
    prefix = {
        "box": "U",
        "r": "R",
        "c": "C",
        "led": "D",
        "d": "D",
        "gnd": "G",
        "v": "V",
        "battary": "BAT",
    }.get(comp_type, "X")
    counters[prefix] += 1
    return f"{prefix}{counters[prefix]}"


def direction_from_body(point: Point, body: Rect) -> Point:
    """Infer the outward orthogonal direction from the component body bounds."""
    x, y = point
    x0, y0, x1, y1 = body
    outside = []
    if x < x0:
        outside.append((x0 - x, (-1, 0)))
    if x > x1:
        outside.append((x - x1, (1, 0)))
    if y < y0:
        outside.append((y0 - y, (0, -1)))
    if y > y1:
        outside.append((y - y1, (0, 1)))
    if outside:
        return max(outside, key=lambda item: item[0])[1]

    return min(
        (
            (abs(x - x0), (-1, 0)),
            (abs(x - x1), (1, 0)),
            (abs(y - y0), (0, -1)),
            (abs(y - y1), (0, 1)),
        ),
        key=lambda item: item[0],
    )[1]


def order_box_pins(box: dict) -> list[str]:
    body = box["bbox"]
    cx = (body[0] + body[2]) / 2
    cy = (body[1] + body[3]) / 2
    keys = list(box["pins"])
    # In image coordinates, atan2(-dy, dx) sorts counterclockwise. Rotate the
    # cycle to the pin nearest the body's upper-left corner.
    keys.sort(key=lambda key: math.atan2(
        -(box["pins"][key]["point"][1] - cy),
        box["pins"][key]["point"][0] - cx,
    ))
    start = min(
        range(len(keys)),
        key=lambda index: (
            abs(box["pins"][keys[index]]["point"][0] - body[0])
            + abs(box["pins"][keys[index]]["point"][1] - body[1])
        ),
    )
    return keys[start:] + keys[:start]


def choose_d_candidate(candidates: list[dict]) -> dict | None:
    usable = [
        candidate for candidate in candidates
        if candidate["type"] != "box"
        and 1 <= len(candidate["info"]["pins"]) <= MAX_PINS_PER_PLACED_COMPONENT
    ]
    if not usable:
        return None
    groups: dict[str, list[dict]] = defaultdict(list)
    for candidate in usable:
        groups[candidate["type"]].append(candidate)
    comp_types = list(groups)
    weights = [D_TYPE_WEIGHTS.get(comp_type, OTHER_D_TYPE_WEIGHT) for comp_type in comp_types]
    chosen_type = random.choices(comp_types, weights=weights, k=1)[0]
    if chosen_type in TYPE_SAMPLE_WEIGHTS:
        pref = TYPE_SAMPLE_WEIGHTS[chosen_type]
        group = groups[chosen_type]
        return random.choices(
            group,
            weights=[pref.get(c["source_name"], 0.05) for c in group],
            k=1,
        )[0]
    return random.choice(groups[chosen_type])


def choose_b_candidate(candidates: list[dict]) -> dict | None:
    usable = [
        candidate for candidate in candidates
        if candidate["type"] != "box" and len(candidate["info"]["pins"]) == 1
    ]
    return random.choice(usable) if usable else None


def place_device_near_pin(candidate: dict, anchor: Point, direction: Point,
                          board_size: tuple[int, int], placed: list[dict],
                          palette_color: tuple[int, int, int], display_name: str):
    board_width, board_height = board_size
    margin_x, margin_y = content_margins(board_width, board_height)
    prepared = prepare_component(candidate, palette_color)
    width, height = prepared["extent_size"]
    local_cx, local_cy = (width - 1) / 2, (height - 1) / 2
    dx, dy = direction
    perpendicular = (-dy, dx)
    for _ in range(MAX_COMPONENT_PLACE_ATTEMPTS):
        distance = random.randint(*DEVICE_OUTWARD_DISTANCE_RANGE)
        side_shift = random.randint(-DEVICE_PERPENDICULAR_JITTER,
                                    DEVICE_PERPENDICULAR_JITTER)
        center_x = anchor[0] + dx * distance + perpendicular[0] * side_shift
        center_y = anchor[1] + dy * distance + perpendicular[1] * side_shift
        x = round(center_x - local_cx)
        y = round(center_y - local_cy)
        extent = (x, y, x + width - 1, y + height - 1)
        if (
            extent[0] < margin_x
            or extent[1] < margin_y
            or extent[2] >= board_width - margin_x
            or extent[3] >= board_height - margin_y
        ):
            continue
        if any(
            rects_overlap(expand_rect(extent, COMPONENT_SAFETY_GAP), old["extent"])
            for old in placed
        ):
            continue
        component = materialize_component(prepared, (x, y), display_name)
        if not component_content_ok(component, board_size):
            continue
        return component
    return None


def place_single_pin_component(candidates: list[dict], anchor: Point,
                               palette_color: tuple[int, int, int],
                               counters: defaultdict[str, int],
                               board_size: tuple[int, int], placed: list[dict]):
    """Place a b-operation device on the box pin without overlapping any bbox."""
    board_width, board_height = board_size
    margin_x, margin_y = content_margins(board_width, board_height)
    candidate_order = list(candidates)
    random.shuffle(candidate_order)
    candidate_order = candidate_order[:B_COMPONENT_CANDIDATE_LIMIT]

    for candidate in candidate_order:
        for _ in range(B_VARIANTS_PER_CANDIDATE):
            prepared = prepare_component(candidate, palette_color)
            pin_key, pin = next(iter(prepared["local_pins"].items()))
            origin = (anchor[0] - pin["point"][0], anchor[1] - pin["point"][1])
            width, height = prepared["extent_size"]
            if (
                origin[0] < margin_x
                or origin[1] < margin_y
                or origin[0] + width >= board_width - margin_x
                or origin[1] + height >= board_height - margin_y
            ):
                continue

            component = materialize_component(prepared, origin, "")
            if not component_content_ok(component, board_size):
                continue
            if any(
                rects_overlap(expand_rect(component["extent"], COMPONENT_SAFETY_GAP), old["extent"])
                for old in placed
            ):
                continue

            component["name"] = make_display_name(candidate["type"], counters)
            # prepare_component() has already rectified this pin by at most
            # 10 px on one axis. The origin is computed from that corrected
            # local point, so the corrected pin lands exactly on the box pin.
            if component["pins"][pin_key]["point"] != anchor:
                continue
            return component
    return None


_FILLER_QUEUE: list[dict] = []
_TYPE_QUEUES: dict[str, list[dict]] = {}


_VARIANT_QUEUE: list[tuple[dict, int]] = []


def choose_filler_variant(candidates: list[dict]):
    """轮转所有 (非 box 样例, 旋转方向)，保证样例和四方向都覆盖。"""
    global _VARIANT_QUEUE
    usable = [c for c in candidates if c["type"] != "box"]
    if not usable:
        return None
    if not _VARIANT_QUEUE:
        _VARIANT_QUEUE = [(c, r) for c in usable for r in ROTATION_STEPS]
        random.shuffle(_VARIANT_QUEUE)
    return _VARIANT_QUEUE.pop(0)


def choose_type_candidate(candidates: list[dict], comp_type: str) -> dict | None:
    usable = [c for c in candidates if c["type"] == comp_type]
    if not usable:
        return None
    if comp_type in TYPE_SAMPLE_WEIGHTS:
        pref = TYPE_SAMPLE_WEIGHTS[comp_type]
        return random.choices(
            usable,
            weights=[pref.get(c["source_name"], 0.05) for c in usable],
            k=1,
        )[0]
    queue = _TYPE_QUEUES.setdefault(comp_type, [])
    if not queue:
        queue.extend(usable)
        random.shuffle(queue)
    return queue.pop(0)


def place_free_component(candidates: list[dict],
                         palette_color: tuple[int, int, int],
                         counters: defaultdict[str, int],
                         board_size: tuple[int, int], placed: list[dict],
                         candidate: dict | None = None,
                         scale_mult: float | None = None, rotation: int | None = None):
    """在画布任意空余位置放一个非 box 器件（不参与网络连接）。"""
    if candidate is None:
        variant = choose_filler_variant(candidates)
        if variant is None:
            return None
        candidate, forced_rotation = variant
        if rotation is None:
            rotation = forced_rotation
    if candidate is None:
        return None
    board_width, board_height = board_size
    margin_x, margin_y = content_margins(board_width, board_height)
    prepared = prepare_component(candidate, palette_color, scale_mult=scale_mult, rotation=rotation)
    width, height = prepared["extent_size"]
    if (
        width + 2 * margin_x >= board_width
        or height + 2 * margin_y >= board_height
    ):
        return None
    for _ in range(MAX_FREE_PLACEMENT_ATTEMPTS):
        x = random.randint(margin_x, board_width - margin_x - width)
        y = random.randint(margin_y, board_height - margin_y - height)
        extent = (x, y, x + width - 1, y + height - 1)
        if any(
            rects_overlap(expand_rect(extent, COMPONENT_SAFETY_GAP), old["extent"])
            for old in placed
        ):
            continue
        component = materialize_component(prepared, (x, y), "")
        if not component_content_ok(component, board_size):
            continue
        component["name"] = make_display_name(candidate["type"], counters)
        return component
    return None


def place_tight_pair(candidates: list[dict], palette_color: tuple[int, int, int],
                    counters: defaultdict[str, int], board_size: tuple[int, int],
                    placed: list[dict], comp_type: str, mirror: bool) -> list[dict]:
    """放一对紧挨的同类符号（case11 风格）。

    LED 用 180° 镜像后长边贴长边；GND/v 保持同向，只留 2-6 px 间隙。
    注意：普通 filler 的安全间距是 30 px，这里刻意跳过，让两枚符号紧贴。
    """
    candidate = choose_type_candidate(candidates, comp_type)
    if candidate is None:
        return []
    scale = next_scale()
    angle = random.choice((0, 90))
    prepared = [
        prepare_component(candidate, palette_color, scale_mult=scale, rotation=angle),
        prepare_component(
            candidate, palette_color, scale_mult=scale,
            rotation=(angle + 180) % 360 if mirror else angle,
        ),
    ]
    b1 = prepared[0]["local_body_bbox"]
    b2 = prepared[1]["local_body_bbox"]
    bw1, bh1 = b1[2] - b1[0], b1[3] - b1[1]
    bw2, bh2 = b2[2] - b2[0], b2[3] - b2[1]
    gap = random.randint(*TIGHT_PAIR_GAP_RANGE)
    bw, bh = board_size
    margin_x, margin_y = content_margins(bw, bh)
    w1, h1 = prepared[0]["extent_size"]
    w2, h2 = prepared[1]["extent_size"]
    # 以真实标注框（body bbox）计算第二枚的 origin，保证两个 GT 框只隔 2-6 px。
    vertical = bh1 >= bw1
    for _ in range(MAX_FREE_PLACEMENT_ATTEMPTS):
        x = random.randint(margin_x, bw - margin_x - max(w1, w2))
        y = random.randint(margin_y, bh - margin_y - max(h1, h2))
        if vertical:
            if random.random() < 0.5:
                oy2 = y + b1[3] + gap - b2[1]
            else:
                oy2 = y + b1[1] - gap - b2[3]
            origins = [(x, y), (x, oy2)]
        else:
            if random.random() < 0.5:
                ox2 = x + b1[2] + gap - b2[0]
            else:
                ox2 = x + b1[0] - gap - b2[2]
            origins = [(x, y), (ox2, y)]
        pair = []
        for prep, origin, name in zip(
            prepared, origins,
            (make_display_name(comp_type, counters), make_display_name(comp_type, counters)),
        ):
            comp = materialize_component(prep, origin, name)
            if not component_content_ok(comp, board_size):
                pair = []
                break
            comp["is_filler"] = True
            pair.append(comp)
        if len(pair) != 2:
            continue
        if any(
            rects_overlap(expand_rect(comp["extent"], COMPONENT_SAFETY_GAP), old["extent"])
            for comp in pair for old in placed
        ):
            continue
        return pair
    return []

def place_led_pair(candidates: list[dict], palette_color: tuple[int, int, int],
                   counters: defaultdict[str, int], board_size: tuple[int, int],
                   placed: list[dict]) -> list[dict]:
    return place_tight_pair(candidates, palette_color, counters, board_size, placed, "led", True)


def place_gnd_pair(candidates: list[dict], palette_color: tuple[int, int, int],
                   counters: defaultdict[str, int], board_size: tuple[int, int],
                   placed: list[dict]) -> list[dict]:
    return place_tight_pair(candidates, palette_color, counters, board_size, placed, "gnd", False)


def make_terminal(component: dict, pin_key: str, board_size: tuple[int, int]):
    point = component["pins"][pin_key]["point"]
    direction = direction_from_body(point, component["bbox"])
    route_start = (
        point[0] + direction[0] * PIN_EXTENSION_LENGTH,
        point[1] + direction[1] * PIN_EXTENSION_LENGTH,
    )
    if not (0 <= route_start[0] < board_size[0] and 0 <= route_start[1] < board_size[1]):
        return None
    return {
        "id": (component["name"], pin_key),
        "component": component["name"],
        "pin_key": pin_key,
        "pinname": component["pins"][pin_key]["pinname"],
        "point": point,
        "route_start": route_start,
        "direction": direction,
    }


# =============================== Graph geometry ===============================

def canonical_edge(a: Point, b: Point) -> Edge:
    return (a, b) if a <= b else (b, a)


def segment_intersection(a: Point, b: Point, c: Point, d: Point):
    """Return None, ('point', point), or ('overlap', None) for orthogonal lines."""
    ab_horizontal = a[1] == b[1]
    cd_horizontal = c[1] == d[1]
    if ab_horizontal and cd_horizontal:
        if a[1] != c[1]:
            return None
        lo = max(min(a[0], b[0]), min(c[0], d[0]))
        hi = min(max(a[0], b[0]), max(c[0], d[0]))
        if lo > hi:
            return None
        if lo == hi:
            return "point", (lo, a[1])
        return "overlap", None
    if not ab_horizontal and not cd_horizontal:
        if a[0] != c[0]:
            return None
        lo = max(min(a[1], b[1]), min(c[1], d[1]))
        hi = min(max(a[1], b[1]), max(c[1], d[1]))
        if lo > hi:
            return None
        if lo == hi:
            return "point", (a[0], lo)
        return "overlap", None
    if ab_horizontal:
        horizontal_a, horizontal_b = a, b
        vertical_a, vertical_b = c, d
    else:
        horizontal_a, horizontal_b = c, d
        vertical_a, vertical_b = a, b
    point = (vertical_a[0], horizontal_a[1])
    if (
        min(horizontal_a[0], horizontal_b[0]) <= point[0] <= max(horizontal_a[0], horizontal_b[0])
        and min(vertical_a[1], vertical_b[1]) <= point[1] <= max(vertical_a[1], vertical_b[1])
    ):
        return "point", point
    return None


def segment_hits_rect(a: Point, b: Point, rect: Rect) -> bool:
    x0, y0, x1, y1 = rect
    if a[1] == b[1]:
        low, high = sorted((a[0], b[0]))
        return y0 < a[1] < y1 and max(low, x0) < min(high, x1)
    if a[0] == b[0]:
        low, high = sorted((a[1], b[1]))
        return x0 < a[0] < x1 and max(low, y0) < min(high, y1)
    raise ValueError(f"非正交线段: {a} -> {b}")


def segments_from_graph(edges: set[Edge], pin_points: set[Point]):
    graph: dict[Point, set[Point]] = defaultdict(set)
    for a, b in edges:
        if a == b:
            continue
        graph[a].add(b)
        graph[b].add(a)

    stops = set(pin_points)
    for point, neighbors in graph.items():
        if len(neighbors) != 2:
            stops.add(point)
            continue
        a, b = tuple(neighbors)
        straight = (
            a[0] == point[0] == b[0]
            or a[1] == point[1] == b[1]
        )
        if not straight:
            stops.add(point)

    visited: set[Edge] = set()
    segments: set[Edge] = set()
    for start in sorted(stops):
        for neighbor in sorted(graph.get(start, ())):
            first_edge = canonical_edge(start, neighbor)
            if first_edge in visited:
                continue
            visited.add(first_edge)
            previous, current = start, neighbor
            while current not in stops:
                following = next(point for point in graph[current] if point != previous)
                visited.add(canonical_edge(current, following))
                previous, current = current, following
            segments.add(canonical_edge(start, current))

    junctions = {point for point, neighbors in graph.items() if len(neighbors) >= 3}
    return sorted(segments), junctions, graph


def graph_is_tree(graph: dict[Point, set[Point]], pin_points: set[Point]) -> bool:
    if not pin_points or not pin_points <= set(graph):
        return False
    start = next(iter(pin_points))
    seen = {start}
    pending = [start]
    edge_count = sum(len(neighbors) for neighbors in graph.values()) // 2
    while pending:
        point = pending.pop()
        for neighbor in graph[point]:
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return pin_points <= seen and len(seen) == len(graph) and edge_count == len(graph) - 1


# ============================== Routing grid / A* ==============================

class RoutingGrid:
    """Nonuniform orthogonal grid: 10 px spacing plus exact pin and obstacle edges."""

    def __init__(self, board_size: tuple[int, int], components: list[dict], terminals: list[dict]):
        self.width, self.height = board_size
        margin_x, margin_y = content_margins(self.width, self.height)
        xs = set(range(margin_x, self.width - margin_x, ROUTE_GRID_SPACING))
        ys = set(range(margin_y, self.height - margin_y, ROUTE_GRID_SPACING))
        xs.add(self.width - margin_x - 1)
        ys.add(self.height - margin_y - 1)
        for terminal in terminals:
            for x, y in (terminal["point"], terminal["route_start"]):
                xs.add(x)
                ys.add(y)
        for component in components:
            x0, y0, x1, y1 = component["extent"]
            xs.update((x0 - 1, x1 + 1))
            ys.update((y0 - 1, y1 + 1))
        self.xs = sorted(x for x in xs if 0 <= x < self.width)
        self.ys = sorted(y for y in ys if 0 <= y < self.height)
        self.x_index = {x: index for index, x in enumerate(self.xs)}
        self.y_index = {y: index for index, y in enumerate(self.ys)}

    def add_points(self, points: Iterable[Point]):
        new_xs = set(self.xs)
        new_ys = set(self.ys)
        for x, y in points:
            if 0 <= x < self.width:
                new_xs.add(x)
            if 0 <= y < self.height:
                new_ys.add(y)
        self.xs = sorted(new_xs)
        self.ys = sorted(new_ys)
        self.x_index = {x: index for index, x in enumerate(self.xs)}
        self.y_index = {y: index for index, y in enumerate(self.ys)}

    def neighbors(self, point: Point):
        x, y = point
        ix = self.x_index[x]
        iy = self.y_index[y]
        if ix > 0:
            yield (self.xs[ix - 1], y), 1
        if ix + 1 < len(self.xs):
            yield (self.xs[ix + 1], y), 1
        if iy > 0:
            yield (x, self.ys[iy - 1]), 2
        if iy + 1 < len(self.ys):
            yield (x, self.ys[iy + 1]), 2


def build_component_obstacles(board_size: tuple[int, int], components: list[dict],
                              routed_terminals: list[dict]):
    width, height = board_size
    alpha_canvas = np.zeros((height, width), dtype=np.uint8)
    for component in components:
        image_alpha = np.asarray(component["image"].getchannel("A"), dtype=np.uint8)
        x, y = component["origin"]
        image_height, image_width = image_alpha.shape
        region = alpha_canvas[y:y + image_height, x:x + image_width]
        np.maximum(region, image_alpha, out=region)

    radius = WIRE_GAP + math.ceil(WIRE_WIDTH / 2)
    expanded = Image.fromarray(alpha_canvas, mode="L").filter(
        ImageFilter.MaxFilter(radius * 2 + 1)
    )
    obstacle_draw = ImageDraw.Draw(expanded)
    for component in components:
        obstacle_draw.rectangle(expand_rect(component["bbox"], radius), fill=255)

    # Pins need a short escape lane from their component body. Clear only the
    # pin's outward lead, extending far enough for the wire to leave the bbox
    # clearance zone; all other approaches remain blocked.
    component_by_name = {component["name"]: component for component in components}
    escape_length = PIN_EXTENSION_LENGTH + radius + 1
    for terminal in routed_terminals:
        component = component_by_name.get(terminal["component"])
        if component is None:
            continue
        point = terminal["point"]
        direction = terminal["direction"]
        escape_end = (
            point[0] + direction[0] * escape_length,
            point[1] + direction[1] * escape_length,
        )
        obstacle_draw.line([point, escape_end], fill=0, width=WIRE_WIDTH)

    obstacle_mask = np.asarray(expanded, dtype=np.uint8) > 0
    return obstacle_mask


def build_route_clearance(board_size: tuple[int, int], routes: list[dict]):
    width, height = board_size
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    clearance_width = 2 * (WIRE_GAP + math.ceil(WIRE_WIDTH / 2)) + 1
    for route in routes:
        for a, b in route["segments"]:
            draw.line([a, b], fill=255, width=clearance_width)
        for x, y in route["junctions"]:
            radius = JUNCTION_RADIUS + WIRE_GAP + math.ceil(WIRE_WIDTH / 2)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    return np.asarray(image, dtype=np.uint8) > 0


def build_foreign_terminal_mask(board_size: tuple[int, int], terminals: list[dict],
                                current_net_ids: set[PinId]):
    width, height = board_size
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    blocked_width = 2 * (WIRE_GAP + math.ceil(WIRE_WIDTH / 2)) + 1
    radius = blocked_width // 2
    for terminal in terminals:
        if terminal["id"] in current_net_ids:
            continue
        draw.line([terminal["point"], terminal["route_start"]], fill=255, width=blocked_width)
        x, y = terminal["route_start"]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    return np.asarray(image, dtype=np.uint8) > 0


def segment_pixel_slice(a: Point, b: Point):
    if a[1] == b[1]:
        low, high = sorted((a[0], b[0]))
        return a[1], slice(low, high + 1)
    if a[0] == b[0]:
        low, high = sorted((a[1], b[1]))
        return slice(low, high + 1), a[0]
    raise ValueError(f"A* 收到非正交线段: {a} -> {b}")


def astar_route(start: Point, goals: set[Point], grid: RoutingGrid,
                blocked: np.ndarray, initial_axis: int = 0):
    if start in goals:
        return [start]
    goals = {goal for goal in goals if 0 <= goal[0] < grid.width and 0 <= goal[1] < grid.height}
    if not goals:
        return None

    gx0 = min(point[0] for point in goals)
    gx1 = max(point[0] for point in goals)
    gy0 = min(point[1] for point in goals)
    gy1 = max(point[1] for point in goals)

    def heuristic(point: Point):
        x, y = point
        return max(gx0 - x, 0, x - gx1) + max(gy0 - y, 0, y - gy1)

    initial_state = (start[0], start[1], initial_axis)
    costs = {initial_state: 0}
    parents = {}
    queue = [(heuristic(start), 0, initial_state)]
    visits = 0

    while queue:
        _, cost, state = heapq.heappop(queue)
        if cost != costs.get(state):
            continue
        point = (state[0], state[1])
        previous_axis = state[2]
        if point in goals:
            path = []
            cursor = state
            while True:
                path.append((cursor[0], cursor[1]))
                if cursor == initial_state:
                    return list(reversed(path))
                cursor = parents[cursor]

        visits += 1
        if visits > MAX_ASTAR_VISITS:
            return None

        for neighbor, axis in grid.neighbors(point):
            row_or_slice = segment_pixel_slice(point, neighbor)
            if blocked[row_or_slice].any():
                continue
            length = abs(point[0] - neighbor[0]) + abs(point[1] - neighbor[1])
            bend = BEND_PENALTY if previous_axis and previous_axis != axis else 0
            next_cost = cost + length + bend
            next_state = (neighbor[0], neighbor[1], axis)
            if next_cost >= costs.get(next_state, float("inf")):
                continue
            costs[next_state] = next_cost
            parents[next_state] = state
            heapq.heappush(queue, (next_cost + heuristic(neighbor), next_cost, next_state))
    return None


def path_edges(path: list[Point]) -> set[Edge]:
    return {canonical_edge(a, b) for a, b in zip(path, path[1:]) if a != b}


def path_conflicts_with_edges(path: list[Point], edges: set[Edge],
                              allowed_points: set[Point]):
    for a, b in zip(path, path[1:]):
        for c, d in edges:
            intersection = segment_intersection(a, b, c, d)
            if intersection is None:
                continue
            kind, point = intersection
            if kind == "overlap":
                return True
            if point not in allowed_points:
                return True
    return False


def path_crosses_routes(path: list[Point], routes: list[dict],
                        crossing: dict | None):
    allowed_point = crossing["point"] if crossing else None
    allowed_route_index = crossing.get("route_index") if crossing else None
    for a, b in zip(path, path[1:]):
        for route_index, route in enumerate(routes):
            for c, d in route["segments"]:
                intersection = segment_intersection(a, b, c, d)
                if intersection is None:
                    continue
                kind, point = intersection
                if kind == "overlap":
                    return True
                if (
                    allowed_point is None
                    or point != allowed_point
                    or route_index != allowed_route_index
                ):
                    return True
    return False


def choose_crossing_candidates(board_size: tuple[int, int], routes: list[dict],
                               obstacle_mask: np.ndarray):
    width, height = board_size
    used = [route["crossing_point"] for route in routes if route.get("crossing_point")]
    candidates = []
    for route_index, route in enumerate(routes):
        for a, b in route["segments"]:
            horizontal = a[1] == b[1]
            length = abs(a[0] - b[0]) if horizontal else abs(a[1] - b[1])
            if length < ROUTE_GRID_SPACING * 4:
                continue
            low, high = sorted((a[0], b[0])) if horizontal else sorted((a[1], b[1]))
            for offset in range(ROUTE_GRID_SPACING * 2, length - ROUTE_GRID_SPACING * 2 + 1,
                                ROUTE_GRID_SPACING):
                value = low + offset
                point = (value, a[1]) if horizontal else (a[0], value)
                approach = (point[0], point[1] - ROUTE_GRID_SPACING) if horizontal else (point[0] - ROUTE_GRID_SPACING, point[1])
                exit_point = (point[0], point[1] + ROUTE_GRID_SPACING) if horizontal else (point[0] + ROUTE_GRID_SPACING, point[1])
                if not all(0 <= p[0] < width and 0 <= p[1] < height for p in (approach, point, exit_point)):
                    continue
                if any(abs(point[0] - old[0]) + abs(point[1] - old[1]) < ROUTE_GRID_SPACING * 2 for old in used):
                    continue
                # Keep the crossing clear of component art and its line width.
                if any(obstacle_mask[p[1], p[0]] for p in (approach, point, exit_point)):
                    continue
                candidates.append({
                    "point": point,
                    "approach": approach,
                    "exit": exit_point,
                    "old_segment": (a, b),
                    "old_horizontal": horizontal,
                    "route_index": route_index,
                })
    random.shuffle(candidates)
    return candidates[:CROSSING_CANDIDATE_LIMIT]


def route_one_net(net_pins: list[dict], all_terminals: list[dict],
                  board_size: tuple[int, int], grid: RoutingGrid,
                  component_obstacles: np.ndarray, existing_routes: list[dict],
                  crossing: dict | None = None):
    if len(net_pins) < 2:
        return None
    net_ids = {terminal["id"] for terminal in net_pins}
    points = {terminal["point"] for terminal in net_pins}
    if len(points) != len(net_pins):
        return None

    base_blocked = component_obstacles | build_foreign_terminal_mask(
        board_size, all_terminals, net_ids
    )
    normal_clearance = build_route_clearance(board_size, existing_routes)
    # A* paths must stay outside existing nets' clearance. Only the explicit
    # straight crossing segment below may pass through that clearance.
    blocked = base_blocked | normal_clearance

    edges: set[Edge] = set()
    for terminal in net_pins:
        if terminal["point"] != terminal["route_start"]:
            edges.add(canonical_edge(terminal["point"], terminal["route_start"]))

    own_edges = set(edges)

    def direction_axis(terminal):
        return 1 if terminal["direction"][0] else 2

    pairs = sorted(
        ((a, b) for index, a in enumerate(net_pins) for b in net_pins[index + 1:]),
        key=lambda pair: (
            abs(pair[0]["route_start"][0] - pair[1]["route_start"][0])
            + abs(pair[0]["route_start"][1] - pair[1]["route_start"][1])
        ),
        reverse=True,
    )
    if crossing:
        pairs = pairs[:CROSSING_ENDPOINT_PAIR_LIMIT]
        cross_point = crossing["point"]
        approach = crossing["approach"]
        exit_point = crossing["exit"]
        grid.add_points((cross_point, approach, exit_point))

        # Prefer endpoint and approach combinations with the shortest initial
        # Manhattan distance, then try the reverse direction as well.
        attempts = []
        for root, second in pairs:
            for entry, leave in ((approach, exit_point), (exit_point, approach)):
                score = (
                    abs(root["route_start"][0] - entry[0])
                    + abs(root["route_start"][1] - entry[1])
                    + abs(second["route_start"][0] - leave[0])
                    + abs(second["route_start"][1] - leave[1])
                )
                attempts.append((score, root, second, entry, leave))
        attempts.sort(key=lambda item: item[0])
        attempts = attempts[:CROSSING_ROUTE_ATTEMPT_LIMIT]
    else:
        attempts = [(0, *pairs[0], None, None)]
        cross_point = None

    branch_blocked = base_blocked | normal_clearance
    for _, root, second, entry, leave in attempts:
        remaining = [terminal for terminal in net_pins if terminal not in (root, second)]
        if crossing:
            # Keep the approach/exit A* searches out of the clearance corridor;
            # the only allowed overlap with the old net is this straight lane.
            middle = [entry, cross_point, leave]
            if any(base_blocked[segment_pixel_slice(a, b)].any()
                   for a, b in zip(middle, middle[1:])):
                continue
            first = astar_route(
                root["route_start"], {entry}, grid, blocked,
                initial_axis=direction_axis(root),
            )
            if first is None or cross_point in first:
                continue
            last = astar_route(
                leave, {second["route_start"]}, grid, blocked,
                initial_axis=2 if crossing["old_horizontal"] else 1,
            )
            if last is None or cross_point in last:
                continue
            trunk = first + middle[1:] + last[1:]
        else:
            trunk = astar_route(
                root["route_start"], {second["route_start"]}, grid, blocked,
                initial_axis=direction_axis(root),
            )
            if trunk is None:
                continue

        if path_crosses_routes(trunk, existing_routes, crossing):
            continue
        if path_conflicts_with_edges(
            trunk, own_edges, {root["route_start"], second["route_start"]}
        ):
            continue

        trial_edges = set(edges)
        trial_edges.update(path_edges(trunk))
        tree_points = set(trunk)
        ordered_remaining = [terminal for terminal in remaining]
        random.shuffle(ordered_remaining)
        valid = True
        for terminal in ordered_remaining:
            start = terminal["route_start"]
            path = astar_route(start, tree_points, grid, branch_blocked,
                               initial_axis=direction_axis(terminal))
            if path is None:
                valid = False
                break
            allowed = {start} | (set(path) & tree_points)
            if path_conflicts_with_edges(path, trial_edges, allowed):
                valid = False
                break
            if path_crosses_routes(path, existing_routes, None):
                valid = False
                break
            trial_edges.update(path_edges(path))
            tree_points.update(path)
        if not valid:
            continue

        segments, junctions, graph = segments_from_graph(trial_edges, points)
        if not graph_is_tree(graph, points) or not segments:
            continue
        return {
            "pin_ids": [terminal["id"] for terminal in net_pins],
            "raw_edges": trial_edges,
            "segments": segments,
            "junctions": junctions,
            "crossing_point": cross_point,
            "crossing_route_index": crossing["route_index"] if crossing else None,
        }
    return None


def point_is_segment_interior(point: Point, segment: Edge) -> bool:
    a, b = segment
    if a[1] == b[1] == point[1]:
        return min(a[0], b[0]) < point[0] < max(a[0], b[0])
    if a[0] == b[0] == point[0]:
        return min(a[1], b[1]) < point[1] < max(a[1], b[1])
    return False


def validate_route_crossings(routes: list[dict], expected_count: int) -> bool:
    """Require only the explicitly marked, interior X crossings between nets."""
    expected: set[tuple[int, int, Point]] = set()
    for route_index, route in enumerate(routes):
        point = route.get("crossing_point")
        other_index = route.get("crossing_route_index")
        if point is None:
            if other_index is not None:
                return False
            continue
        if other_index is None or not (0 <= other_index < route_index):
            return False
        if point in route.get("junctions", ()):
            return False
        if point in routes[other_index].get("junctions", ()):
            return False
        expected.add((other_index, route_index, point))
    if len(expected) != expected_count:
        return False

    found: set[tuple[int, int, Point]] = set()
    for first_index, first in enumerate(routes):
        for second_index in range(first_index + 1, len(routes)):
            second = routes[second_index]
            for a, b in first["segments"]:
                for c, d in second["segments"]:
                    intersection = segment_intersection(a, b, c, d)
                    if intersection is None:
                        continue
                    kind, point = intersection
                    if kind == "overlap":
                        return False
                    crossing_key = (first_index, second_index, point)
                    if crossing_key not in expected:
                        return False
                    if not point_is_segment_interior(point, (a, b)):
                        return False
                    if not point_is_segment_interior(point, (c, d)):
                        return False
                    if (a[1] == b[1]) == (c[1] == d[1]):
                        return False
                    found.add(crossing_key)
    return found == expected and len(found) == expected_count


def get_noncrossing_wired_nets(pin_nets: list[dict], all_terminals: list[dict],
                               board_size: tuple[int, int], components: list[dict],
                               crossing_count: int):
    def routing_priority(net: dict):
        starts = [terminal["route_start"] for terminal in net["terminals"]]
        span = max(
            (abs(a[0] - b[0]) + abs(a[1] - b[1])
             for index, a in enumerate(starts) for b in starts[index + 1:]),
            default=0,
        )
        return span * math.sqrt(len(starts))

    # Route the widest networks first. Their longer segments provide usable
    # crossing candidates for later independent nets.
    pin_nets = sorted(pin_nets, key=routing_priority, reverse=True)
    routed_terminals = [
        terminal for net in pin_nets for terminal in net["terminals"]
    ]
    grid = RoutingGrid(board_size, components, routed_terminals)
    obstacle_mask = build_component_obstacles(board_size, components, routed_terminals)
    if not pin_nets:
        return None

    requested = min(crossing_count, max(0, len(pin_nets) - 1))
    if requested and len(pin_nets) < 2:
        return None
    crossed = 0
    routed = []
    for index, net in enumerate(pin_nets):
        crossing_for_net = None
        needs_crossing = index > 0 and crossed < requested
        if needs_crossing:
            candidates = choose_crossing_candidates(board_size, routed, obstacle_mask)
            for candidate in candidates:
                trial = route_one_net(
                    net["terminals"], all_terminals, board_size, grid,
                    obstacle_mask, routed, crossing=candidate,
                )
                if trial is not None:
                    crossing_for_net = candidate
                    routed.append(trial)
                    crossed += 1
                    break
            if crossing_for_net is None:
                return None
        else:
            trial = route_one_net(
                net["terminals"], all_terminals, board_size, grid,
                obstacle_mask, routed,
            )
            if trial is None:
                return None
            routed.append(trial)
    if crossed != requested:
        return None
    return routed


# =========================== Circuit construction ============================

def add_terminal(component: dict, pin_key: str, board_size: tuple[int, int],
                 terminals: dict[PinId, dict]):
    terminal = make_terminal(component, pin_key, board_size)
    if terminal is None:
        return None
    component["pins"][pin_key]["terminal"] = terminal
    terminals[terminal["id"]] = terminal
    return terminal


def attempt_layout(candidates: list[dict]):
    board_size = (
        random.randint(*BOARD_WIDTH_RANGE),
        random.randint(*BOARD_HEIGHT_RANGE),
    )
    palette_name, palette_color, wire_color = choose_palette()
    box_candidate = choose_box_candidate(candidates)
    box = pick_box_component(box_candidate, board_size, palette_color)
    if box is None:
        return None
    components = [box]
    counters: defaultdict[str, int] = defaultdict(int)
    counters["U"] = 1
    terminals: dict[PinId, dict] = {}
    pin_nets: list[dict] = []
    open_pins: set[PinId] = set()
    consumed: set[str] = set()
    ordered_keys = order_box_pins(box)[:MAX_BOX_PINS_USED]
    single_pin_candidates = [
        candidate for candidate in candidates
        if candidate["type"] != "box" and len(candidate["info"]["pins"]) == 1
    ]

    index = 0
    while index < len(ordered_keys):
        box_pin_key = ordered_keys[index]
        if box_pin_key in consumed:
            index += 1
            continue
        box_pin = box["pins"][box_pin_key]
        pin_id = (box["name"], box_pin_key)
        roll = random.random()

        # a: preserve the pin and draw an open marker.
        if roll < 0.20:
            box_pin["open"] = True
            open_pins.add(pin_id)
            index += 1
            continue

        # b: attach a single-pin component to this box pin without bbox overlap.
        if roll < 0.30:
            attached = place_single_pin_component(
                single_pin_candidates,
                box_pin["point"],
                palette_color,
                counters,
                board_size,
                components,
            )
            if attached is not None:
                components.append(attached)
                attached_key = next(iter(attached["pins"]))
                box_pin["action"] = "b"
                attached["pins"][attached_key]["action"] = "b"
                box_terminal = {
                    "id": pin_id,
                    "component": box["name"],
                    "pin_key": box_pin_key,
                    "pinname": box_pin["pinname"],
                    "point": box_pin["point"],
                }
                attached_terminal = {
                    "id": (attached["name"], attached_key),
                    "component": attached["name"],
                    "pin_key": attached_key,
                    "pinname": attached["pins"][attached_key]["pinname"],
                    "point": box_pin["point"],
                }
                pin_nets.append({
                    "kind": "b",
                    "terminals": [box_terminal, attached_terminal],
                })
                attached["pins"][attached_key]["net_kind"] = "b"
                box_pin["net_kind"] = "b"
                index += 1
                continue

            # No legal b placement exists. v2 explicitly turns this into a.
            box_pin["open"] = True
            open_pins.add(pin_id)
            box_pin["action"] = "a"
            index += 1
            continue

        # c: pair this box pin with the next unprocessed one. A final pin has
        # no successor and falls through to d.
        if roll < 0.50 and index + 1 < len(ordered_keys):
            next_key = ordered_keys[index + 1]
            if next_key not in consumed:
                first_terminal = add_terminal(box, box_pin_key, board_size, terminals)
                second_terminal = add_terminal(box, next_key, board_size, terminals)
                if first_terminal is None or second_terminal is None:
                    return None
                box_pin["action"] = "c"
                box["pins"][next_key]["action"] = "c_skipped"
                box_pin["net_kind"] = "c"
                box["pins"][next_key]["net_kind"] = "c"
                consumed.add(next_key)
                pin_nets.append({"kind": "c", "terminals": [first_terminal, second_terminal]})
                index += 1
                continue

        # d: place one or two preferred non-box components in the free sector
        # associated with the anchor pin. Failed placement follows the task's
        # fallback: leave this pin open instead of dropping it silently.
        anchor_terminal = add_terminal(box, box_pin_key, board_size, terminals)
        if anchor_terminal is None:
            return None
        placed_for_net = []
        requested_devices = random.choices(
            DEVICES_PER_D_ACTION,
            weights=(1.0 - SECOND_DEVICE_PROBABILITY, SECOND_DEVICE_PROBABILITY),
            k=1,
        )[0]
        for _ in range(requested_devices):
            candidate = choose_d_candidate(candidates)
            if candidate is None:
                break
            name = make_display_name(candidate["type"], counters)
            component = place_device_near_pin(
                candidate,
                box_pin["point"],
                anchor_terminal["direction"],
                board_size,
                components,
                palette_color,
                name,
            )
            if component is None:
                continue
            components.append(component)
            placed_for_net.append(component)

        d_terminals = [anchor_terminal]
        for component in placed_for_net:
            for pin_key in component["pins"]:
                terminal = add_terminal(component, pin_key, board_size, terminals)
                if terminal is None:
                    return None
                d_terminals.append(terminal)
        if len(d_terminals) < 2:
            # Avoid a dangling unmarked box pin if all placements failed.
            box_pin.pop("terminal", None)
            terminals.pop(pin_id, None)
            box_pin["open"] = True
            open_pins.add(pin_id)
        else:
            box_pin["action"] = "d"
            box_pin["net_kind"] = "d"
            pin_nets.append({"kind": "d", "terminals": d_terminals})
        index += 1

    # Repeated source labels are allowed; annotation identities use component
    # display names plus source pin keys, which remain unique.
    routed_nets = [net for net in pin_nets if net["kind"] in ("c", "d")]
    if len(routed_nets) < 2:
        return None
    crossing_count = min(
        random.randint(*CROSSING_NET_COUNT_RANGE), len(routed_nets) - 1
    )

    routes = get_noncrossing_wired_nets(
        routed_nets,
        list(terminals.values()),
        board_size,
        components,
        crossing_count,
    )
    if routes is None or not validate_route_crossings(routes, crossing_count):
        return None

    # Associate routed paths with their hypergraphs while retaining b nets with
    # empty edges in the annotation.
    route_by_pinset = {frozenset(route["pin_ids"]): route for route in routes}
    for net in pin_nets:
        if net["kind"] in ("c", "d"):
            key = frozenset(terminal["id"] for terminal in net["terminals"])
            net["route"] = route_by_pinset[key]

    # 在已布好的网络之外补充随机空余器件，使每个用例的器件数落在 10~50。
    # 这些器件只写 components/pins，不加入任何网络。
    if len(components) > MAX_COMPONENTS_PER_SAMPLE:
        return None
    # 先放 case11 风格的紧贴对，避免被后续最低配额占满位置。
    # LED：两枚相差 180°、长边相邻（2-6 px 间隙）。
    for _ in range(random.randint(*LED_PAIR_COUNT_RANGE)):
        if len(components) + 2 > MAX_COMPONENTS_PER_SAMPLE:
            break
        pair = place_led_pair(candidates, palette_color, counters, board_size, components)
        if not pair:
            continue
        components.extend(pair)
    # GND：同向、2-6 px 间隙，模拟 case11 的 ∅346/∅695。
    for _ in range(random.randint(*GND_PAIR_COUNT_RANGE)):
        if len(components) + 2 > MAX_COMPONENTS_PER_SAMPLE:
            break
        pair = place_gnd_pair(candidates, palette_color, counters, board_size, components)
        if not pair:
            continue
        components.extend(pair)
    # 再保证低检测率类别的最低出现次数（同一张图里多放几个冷门样例）。
    for cold_type, minimum in COLD_MIN_COUNTS.items():
        while (
            sum(1 for c in components if c["type"] == cold_type) < minimum
            and len(components) < MAX_COMPONENTS_PER_SAMPLE
        ):
            cand = choose_type_candidate(candidates, cold_type)
            if cand is None:
                break
            filler = place_free_component(
                candidates, palette_color, counters, board_size, components,
                candidate=cand,
            )
            if filler is None:
                break
            filler["is_filler"] = True
            components.append(filler)
    # v/gnd/led：每张图都补齐 0/90/180/270 四个方向。
    for cold_type in ("v", "gnd", "led"):
        existing = {c.get("rotation") for c in components if c["type"] == cold_type}
        for angle in ROTATION_STEPS:
            if angle in existing or len(components) >= MAX_COMPONENTS_PER_SAMPLE:
                continue
            cand = choose_type_candidate(candidates, cold_type)
            if cand is None:
                break
            filler = place_free_component(
                candidates, palette_color, counters, board_size, components,
                candidate=cand, rotation=angle,
            )
            if filler is None:
                continue
            filler["is_filler"] = True
            components.append(filler)
            existing.add(angle)
    if len(components) > MAX_COMPONENTS_PER_SAMPLE:
        return None
    target_components = max(
        len(components),
        random.randint(MIN_COMPONENTS_PER_SAMPLE, MAX_COMPONENTS_PER_SAMPLE),
    )
    target_components = min(target_components, MAX_COMPONENTS_PER_SAMPLE)
    while len(components) < target_components:
        filler = place_free_component(
            candidates, palette_color, counters, board_size, components
        )
        if filler is None:
            break
        filler["is_filler"] = True
        components.append(filler)
    if len(components) < MIN_COMPONENTS_PER_SAMPLE:
        return None

    return {
        "board_size": board_size,
        "palette": palette_name,
        "component_color": palette_color,
        "wire_color": wire_color,
        "components": components,
        "terminals": terminals,
        "pin_nets": pin_nets,
        "open_pins": open_pins,
        "routes": routes,
        "noise_wires": ensure_component_wires(
            components, pin_nets,
            build_noise_wires(components, pin_nets, board_size), board_size
        ),
    }


# ================================= Rendering ==================================

def build_noise_wires(components: list[dict], pin_nets: list[dict],
                      board_size: tuple[int, int]) -> list[list[Point]]:
    """给每个尚未接入 GT 网络的元件补一条完整折线，绕开所有元件 bbox。

    每个未连接元件取一个代表引脚，和最近的另一个元件代表引脚相连。
    返回多条 polyline；渲染时每条线用统一线宽和圆角连接，避免断断续续。
    """
    connected = {
        tuple(terminal["id"])
        for net in pin_nets
        for terminal in net.get("terminals", [])
    }
    connected_comps = {component for component, _ in connected}
    boxes = [component["bbox"] for component in components]
    terms = []
    for component in components:
        if component["name"] in connected_comps or not component["pins"]:
            continue
        pin_key = next(iter(component["pins"]))
        terminal = make_terminal(component, pin_key, board_size)
        if terminal is None:
            continue
        direction = terminal["direction"]
        distance = max(PIN_EXTENSION_LENGTH, 20)
        while distance < 120 and any(
            segment_hits_rect(terminal["point"], terminal["route_start"], box)
            for box in boxes
        ):
            distance += 2
            terminal["route_start"] = (
                terminal["point"][0] + direction[0] * distance,
                terminal["point"][1] + direction[1] * distance,
            )
        if any(segment_hits_rect(terminal["point"], terminal["route_start"], box) for box in boxes):
            continue
        if not (0 <= terminal["route_start"][0] < board_size[0]
                and 0 <= terminal["route_start"][1] < board_size[1]):
            continue
        terms.append(terminal)
    if len(terms) < 2:
        return []

    grid = RoutingGrid(board_size, components, terms)
    blocked = build_component_obstacles(board_size, components, [])
    lines: list[list[Point]] = []

    def clean(points: list[Point]) -> list[Point]:
        out = []
        for point in points:
            if not out or point != out[-1]:
                out.append(point)
        return out

    def l_path_clear(a: Point, b: Point) -> list[Point] | None:
        for corner in ((a[0], b[1]), (b[0], a[1])):
            path = clean([a, corner, b])
            if all(
                not any(segment_hits_rect(p, q, box) for box in boxes)
                for p, q in zip(path, path[1:])
            ):
                return path
        return None

    def stubs_ok(a, b) -> bool:
        return not any(
            segment_hits_rect(p, q, box)
            for p, q in ((a["point"], a["route_start"]), (b["route_start"], b["point"]))
            for box in boxes
        )

    def connect(a, b) -> bool:
        if not stubs_ok(a, b):
            return False
        path = astar_route(a["route_start"], {b["route_start"]}, grid, blocked)
        if path:
            lines.append(clean([a["point"], a["route_start"]] + path + [b["point"]]))
            return True
        fallback = l_path_clear(a["route_start"], b["route_start"])
        if fallback is not None:
            lines.append(clean([a["point"], a["route_start"]] + fallback + [b["point"]]))
            return True
        return False

    remaining = list(range(len(terms)))
    while len(remaining) >= 2:
        i = remaining.pop(0)
        order = sorted(
            remaining,
            key=lambda j: (terms[j]["route_start"][0] - terms[i]["route_start"][0]) ** 2
            + (terms[j]["route_start"][1] - terms[i]["route_start"][1]) ** 2,
        )
        for j in order:
            if connect(terms[i], terms[j]):
                remaining.remove(j)
                break
    if remaining:
        i = remaining[0]
        order = sorted(
            (j for j in range(len(terms)) if j != i),
            key=lambda j: (terms[j]["route_start"][0] - terms[i]["route_start"][0]) ** 2
            + (terms[j]["route_start"][1] - terms[i]["route_start"][1]) ** 2,
        )
        for j in order:
            if connect(terms[i], terms[j]):
                break
    # 极端拥挤时给还没连上的元件一条向外短引线。
    endpoints = {p for line in lines for p in line}
    for term in terms:
        if term["point"] in endpoints:
            continue
        direction = term["direction"]
        stub = (
            term["point"][0] + direction[0] * 24,
            term["point"][1] + direction[1] * 24,
        )
        if not any(segment_hits_rect(term["point"], stub, box) for box in boxes):
            lines.append([term["point"], stub])
    return lines

def ensure_component_wires(components: list[dict], pin_nets: list[dict],
                           wires: list[list[Point]],
                           board_size: tuple[int, int]) -> list[list[Point]]:
    """兜底：任何仍未出现在 GT 网络/噪声线端点上的元件补一条向外短引线。"""
    connected = {
        tuple(terminal["id"])
        for net in pin_nets
        for terminal in net.get("terminals", [])
    }
    endpoints = {p for seg in wires for p in seg}
    boxes = [component["bbox"] for component in components]
    all_pins = [
        (component, pin_key, pin["point"])
        for component in components
        for pin_key, pin in component["pins"].items()
    ]
    for component in components:
        if not component["pins"]:
            continue
        if any(
            (component["name"], key) in connected or pin["point"] in endpoints
            for key, pin in component["pins"].items()
        ):
            continue
        # 先尝试连到非常近的另一个元件引脚（例如 LED 对中间的 2~10px 间隙）。
        best = None
        for key, pin in component["pins"].items():
            point = pin["point"]
            for other, other_key, other_point in all_pins:
                if other["name"] == component["name"]:
                    continue
                distance = math.hypot(point[0] - other_point[0], point[1] - other_point[1])
                if distance <= 14 and (best is None or distance < best[0]):
                    best = (distance, point, other_point)
        if best is not None:
            _, point, other_point = best
            # 紧贴对（LED/GND）的引脚可能不对齐：只允许横平竖直，斜连会触发
            # segment_hits_rect 的“非正交线段”异常。用 L 形折线连接。
            if point[0] == other_point[0] or point[1] == other_point[1]:
                path = [point, other_point]
            else:
                candidates = ([point, (other_point[0], point[1]), other_point],
                              [point, (point[0], other_point[1]), other_point])
                path = None
                for candidate in candidates:
                    if all(
                        not segment_hits_rect(a, b, box)
                        for a, b in zip(candidate, candidate[1:])
                        for box in boxes
                    ):
                        path = candidate
                        break
            if path is not None and all(
                not segment_hits_rect(a, b, box)
                for a, b in zip(path, path[1:]) for box in boxes
            ):
                wires.append(path)
                continue
        added = False
        for pin in component["pins"].values():
            point = pin["point"]
            for direction in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                stub = (point[0] + direction[0] * 24, point[1] + direction[1] * 24)
                if not (0 <= stub[0] < board_size[0] and 0 <= stub[1] < board_size[1]):
                    continue
                if any(segment_hits_rect(point, stub, box) for box in boxes):
                    continue
                wires.append([point, stub])
                added = True
                break
            if added:
                break
    return wires


def draw_nearby_texts(board: Image.Image, sample: dict) -> None:
    """为每个元件在 bbox 外部写一个较小的 type 文字，绝不压住自身 bbox。"""
    width, height = sample["board_size"]
    margin_x, margin_y = content_margins(width, height)
    draw = ImageDraw.Draw(board)
    all_boxes = [component["bbox"] for component in sample["components"]]

    def overlaps(rect, boxes):
        x0, y0, x1, y1 = rect
        return any(
            x0 < b[2] and x1 > b[0] and y0 < b[3] and y1 > b[1]
            for b in boxes
        )

    def try_place(component, font, check_all):
        text = str(component["type"])
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw + 2 * margin_x >= width or th + 2 * margin_y >= height:
            return None
        bx0, by0, bx1, by1 = component["bbox"]
        sides = ["top", "bottom", "left", "right"]
        random.shuffle(sides)
        for side in sides:
            gap = random.randint(*NEARBY_TEXT_GAP_RANGE)
            if side == "top":
                x, y = random.randint(bx0, bx1), by0 - th - gap
            elif side == "bottom":
                x, y = random.randint(bx0, bx1), by1 + gap
            elif side == "left":
                x, y = bx0 - tw - gap, random.randint(by0, by1)
            else:
                x, y = bx1 + gap, random.randint(by0, by1)
            x = min(max(margin_x, x), width - margin_x - tw - 1)
            y = min(max(margin_y, y), height - margin_y - th - 1)
            rect = (x, y, x + tw, y + th)
            if overlaps(rect, [component["bbox"]]):
                continue
            if check_all and overlaps(rect, all_boxes):
                continue
            return rect, font
        return None

    records = []
    for component in sample["components"]:
        placed = None
        used_font = None
        # 先尽量避开所有元件 bbox；放不下时退化为只避开自身 bbox。
        for check_all in (True, False):
            for size in (random.randint(*NEARBY_TEXT_FONT_RANGE), 14, 12, 10, 8):
                font = _load_font(size)
                placed = try_place(component, font, check_all)
                if placed:
                    placed, used_font = placed
                    break
            if placed:
                break
        if not placed:
            continue
        text = str(component["type"])
        draw.text((placed[0], placed[1]), text, font=used_font, fill=(0, 0, 0))
        records.append({"component": component["name"], "text": text, "bbox": list(placed)})
    sample["noise_texts"] = records

def render_sample(sample: dict) -> Image.Image:
    width, height = sample["board_size"]
    color = sample["wire_color"]
    component_color = sample["component_color"]
    board = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(board)

    # Draw routed nets before symbols so pin and component art remains crisp.
    for route in sample["routes"]:
        for a, b in route["segments"]:
            draw.line([a, b], fill=color, width=WIRE_WIDTH)
        for x, y in route["junctions"]:
            draw.ellipse(
                (x - JUNCTION_RADIUS, y - JUNCTION_RADIUS,
                 x + JUNCTION_RADIUS, y + JUNCTION_RADIUS),
                fill=component_color,
            )
    # 模拟连线画在元件之后：路径已绕开 bbox，这样不会被元件盖断。
    for line in sample.get("noise_wires", []):
        if len(line) >= 2:
            draw.line(
                line,
                fill=color,
                width=random.randint(*NOISE_WIRE_WIDTH_RANGE),
                joint="curve",
            )

    for component in sample["components"]:
        board.paste(component["image"], component["origin"], component["image"])

    # 干扰线画在器件之上（可穿过 PNG），再补器件附近的黑色随机文字。
    draw_nearby_texts(board, sample)

    draw = ImageDraw.Draw(board)
    for pin_id in sample["open_pins"]:
        component_name, pin_key = pin_id
        component = next(comp for comp in sample["components"] if comp["name"] == component_name)
        x, y = component["pins"][pin_key]["point"]
        marker = random.choice(OPEN_MARKER_CHOICES)
        if marker == "dot":
            radius = OPEN_MARKER_RADIUS
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
            )
        else:
            cross = OPEN_MARKER_CROSS_HALF_SIZE
            draw.line(
                (x - cross, y - cross, x + cross, y + cross),
                fill=color,
                width=OPEN_MARKER_CROSS_WIDTH,
            )
            draw.line(
                (x - cross, y + cross, x + cross, y - cross),
                fill=color,
                width=OPEN_MARKER_CROSS_WIDTH,
            )
    return board


# =============================== JSON annotation ===============================

def point_to_json(point: Point, board_height: int):
    """Serialize a pixel point after reflection around the canvas centre x-axis."""
    return {
        "x": int(point[0]),
        "y": int(board_height - point[1]),
    }


def edge_list_for_annotation(route: dict, board_height: int):
    return {
        f"edge_{index}": [
            point_to_json(a, board_height),
            point_to_json(b, board_height),
        ]
        for index, (a, b) in enumerate(route["segments"], start=1)
    }


def build_annotation(sample: dict):
    components = {}
    pins = {}
    nets = {}
    board_height = sample["board_size"][1]
    for component in sample["components"]:
        x0, y0, x1, y1 = component["bbox"]
        components[component["name"]] = {
            "Name": component["name"],
            "type": component["type"],
            "value": None,
            "bbox": [
                int(x0),
                int(board_height - y1),
                int(x1),
                int(board_height - y0),
            ],
        }
        pins[component["name"]] = {}
        for pin_key, pin in component["pins"].items():
            pins[component["name"]][pin_key] = {
                "pinname": pin["pinname"],
                "point": point_to_json(pin["point"], board_height),
            }

    for index, net in enumerate(sample["pin_nets"], start=1):
        refs = []
        for terminal in net["terminals"]:
            pin_key = str(terminal["pin_key"])
            if pin_key.startswith("pin_"):
                pin_key = pin_key[len("pin_"):]
            refs.append(f"{terminal['component']}.{pin_key}")
        route = net.get("route")
        nets[f"net_{index}"] = {
            "hyperGraph": "(" + ",".join(refs) + ")",
            "edges": edge_list_for_annotation(route, board_height) if route else {},
        }

    return {"components": components, "pins": pins, "nets": nets}


# ================================== Entrypoint =================================

def generate_one_sample(sample_id: int, candidates: list[dict], *, verbose: bool = True):
    if RANDOM_SEED is not None:
        sample_seed = (RANDOM_SEED + sample_id * 1_000_003) % (2 ** 32)
        random.seed(sample_seed)
        np.random.seed(sample_seed)

    for attempt in range(1, MAX_LAYOUT_ATTEMPTS + 1):
        sample = attempt_layout(candidates)
        if sample is None:
            continue
        component_count = len(sample["components"])
        if not (MIN_COMPONENTS_PER_SAMPLE <= component_count <= MAX_COMPONENTS_PER_SAMPLE):
            continue
        board = render_sample(sample)
        annotation = build_annotation(sample)
        target_dir = OUTPUT_DIR / f"{sample_id:04d}"
        if target_dir.exists():
            return False
        temporary_dir = Path(tempfile.mkdtemp(prefix=f".{sample_id:04d}-", dir=OUTPUT_DIR))
        try:
            board.save(temporary_dir / "circuit.png", format="PNG")
            with (temporary_dir / "annotation.json").open("w", encoding="utf-8") as handle:
                json.dump(annotation, handle, ensure_ascii=False, indent=2)
            if WRITE_DEBUG:
                debug = {
                    "components": [
                        {
                            "name": c["name"], "type": c["type"],
                            "source_name": c["source_name"],
                            "rotation": c.get("rotation"),
                            "scale_mult": c.get("scale_mult"),
                            "is_filler": bool(c.get("is_filler")),
                        }
                        for c in sample["components"]
                    ],
                    "noise_wires": sample.get("noise_wires", []),
                    "noise_texts": sample.get("noise_texts", []),
                    "net_pins": [
                        list(pair) for pair in sorted({
                            (t["component"], t["pin_key"])
                            for net in sample["pin_nets"]
                            for t in net["terminals"]
                        })
                    ],
                }
                with (temporary_dir / "debug.json").open("w", encoding="utf-8") as handle:
                    json.dump(debug, handle, ensure_ascii=False, indent=2)
            os.replace(temporary_dir, target_dir)
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)
        if verbose:
            print(
                f"[{sample_id:04d}] 完成: 尝试 {attempt}, "
                f"画布 {sample['board_size'][0]}x{sample['board_size'][1]}, "
                f"器件 {len(sample['components'])}, 网络 {len(sample['pin_nets'])}",
                flush=True,
            )
        return True
    if verbose:
        print(f"[{sample_id:04d}] {MAX_LAYOUT_ATTEMPTS} 次布局与布线均未成功", flush=True)
    return False


def _init_generation_worker(candidates: list[dict], progress_queue):
    global _WORKER_CANDIDATES, _WORKER_PROGRESS_QUEUE
    _WORKER_CANDIDATES = candidates
    _WORKER_PROGRESS_QUEUE = progress_queue


def _generate_sample_chunk(sample_ids: list[int]):
    """Generate a statically assigned, balanced chunk in a worker process."""
    if _WORKER_CANDIDATES is None or _WORKER_PROGRESS_QUEUE is None:
        raise RuntimeError("生成进程未正确初始化")
    for sample_id in sample_ids:
        success = False
        error_message = ""
        try:
            success = generate_one_sample(sample_id, _WORKER_CANDIDATES, verbose=False)
        except Exception as error:
            error_message = f"[{sample_id:04d}] 生成异常: {type(error).__name__}: {error}"
        _WORKER_PROGRESS_QUEUE.put((sample_id, success, error_message))


def main():
    if ROUTE_GRID_SPACING <= 0 or WIRE_WIDTH <= 0:
        raise ValueError("路由网格间隔和线宽必须大于 0")
    if NUM_SAMPLES < 0:
        raise ValueError("NUM_SAMPLES 不能小于 0")
    if PROCESS_COUNT <= 0:
        raise ValueError("PROCESS_COUNT 必须大于 0")
    if len(ROTATION_ANGLES) != len(ROTATION_WEIGHTS):
        raise ValueError("旋转角度数量与权重数量不一致")
    if not 0 <= SECOND_DEVICE_PROBABILITY <= 1:
        raise ValueError("SECOND_DEVICE_PROBABILITY 必须在 0 到 1 之间")

    pins_info = load_pins_info(PINS_JSON_PATH)
    files_by_type = scan_component_pngs(BASE_COMPS_DIR)
    candidates = collect_candidates(files_by_type, pins_info)
    boxes = [candidate for candidate in candidates if candidate["type"] == "box"]
    if not boxes:
        raise RuntimeError("没有找到带 pin 元数据的 box 元件")
    if not any(
        candidate["type"] != "box" and len(candidate["info"]["pins"]) == 1
        for candidate in candidates
    ):
        print("提示: 没有单引脚器件，b 操作会回退到 a 操作", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"扫描完成: {len(files_by_type)} 种类型, {len(candidates)} 个有效元件, "
        f"输出目录 {OUTPUT_DIR}",
        flush=True,
    )
    existing_ids = {
        int(path.name)
        for path in OUTPUT_DIR.iterdir()
        if path.is_dir() and path.name.isdigit()
    }
    sample_count = len(existing_ids)
    missing_count = max(0, NUM_SAMPLES - sample_count)
    sample_ids = []
    sample_id = 1
    while len(sample_ids) < missing_count:
        if sample_id not in existing_ids:
            sample_ids.append(sample_id)
        sample_id += 1

    if sample_ids:
        worker_count = min(PROCESS_COUNT, len(sample_ids))
        chunks = [sample_ids[index::worker_count] for index in range(worker_count)]
        succeeded = 0

        if worker_count == 1:
            with tqdm(total=len(sample_ids), desc="生成样本", unit="个") as progress:
                for current_id in chunks[0]:
                    try:
                        generated = generate_one_sample(current_id, candidates, verbose=False)
                    except Exception as error:
                        generated = False
                        tqdm.write(
                            f"[{current_id:04d}] 生成异常: {type(error).__name__}: {error}"
                        )
                    succeeded += int(generated)
                    progress.update(1)
        else:
            context = multiprocessing.get_context("spawn")
            progress_queue = context.Queue()
            pool = context.Pool(
                processes=worker_count,
                initializer=_init_generation_worker,
                initargs=(candidates, progress_queue),
            )
            try:
                jobs = [pool.apply_async(_generate_sample_chunk, (chunk,)) for chunk in chunks]
                pool.close()
                with tqdm(total=len(sample_ids), desc="生成样本", unit="个") as progress:
                    for _ in sample_ids:
                        current_id, generated, error_message = progress_queue.get()
                        succeeded += int(generated)
                        if error_message:
                            tqdm.write(error_message)
                        progress.update(1)
                for job in jobs:
                    job.get()
                pool.join()
            except BaseException:
                pool.terminate()
                pool.join()
                raise
            finally:
                progress_queue.close()
                progress_queue.join_thread()

        sample_count += succeeded
        print(
            f"本次成功生成 {succeeded}/{len(sample_ids)} 个样本；"
            f"Synthetic_Data 中共有 {sample_count}/{NUM_SAMPLES} 个编号样本",
            flush=True,
        )
    else:
        print(f"已达到目标数量: {sample_count}/{NUM_SAMPLES}", flush=True)


if __name__ == "__main__":
    main()
