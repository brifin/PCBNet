import bisect
import heapq
import json
import math
import random
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ===================== 配置 =====================

BASE_DIR = Path("/data/luoxuan/PCBNet/data/base_comps")
PINS_JSON_PATH = BASE_DIR / "pins.json"
OUTPUT_DIR = Path("/data/luoxuan/PCBNet/data/Synthetic_Data")

NUM_CASES = 300 # 生成数量

# 网络随机性设置
BOARD_W = 2000 # 宽
BOARD_H = 1500 # 高
SAMPLE_COUNT = 5 # 器件数量
MAX_NET_SIZE = 5 # 一条 net（网络）里面包含的引脚 pin 数量的上限

def rand_net_params():
    global BOARD_W, BOARD_H, SAMPLE_COUNT, MIN_NET_SIZE, MAX_NET_SIZE
    BOARD_W = random.randint(1600, 2500)
    BOARD_H = random.randint(1200, 2000)
    SAMPLE_COUNT = random.randint(3, 10)
    MAX_NET_SIZE = random.randint(3, 8)
    print(f"BOARD_W={BOARD_W}, BOARD_H={BOARD_H}, SAMPLE_COUNT={SAMPLE_COUNT}, MAX_NET_SIZE={MAX_NET_SIZE}")

# 默认配置
MIN_NET_SIZE = 2 
BASE_H = 150 # 基准高度
HEIGHT_JITTER = (0.85, 1.15) # 高度抖动
COMPONENT_MARGIN = 65
BOARD_MARGIN = 70
ROUTE_GRID = 20
WIRE_WIDTH = 4
WIRE_GAP = 10
JUNCTION_RADIUS = 6
BEND_PENALTY = 100
# 每个 pin 至少向外直线延伸这么多像素。
# 最终长度可能稍大，以便末端落在对应方向的网格坐标上。
PIN_EXTENSION_LENGTH = 20
MAX_LAYOUT_TRIES = 100
MAX_COMPONENT_PLACE_TRIES = 500
MAX_ASTAR_STATES = 400000
ROTATION_CHOICES = (0, 90, 180, 270)
ROTATION_WEIGHTS = (0.6, 0.1, 0.2, 0.1)
RANDOM_SEED = None

FONT_PATH = None
PIN_FONT_SIZE_RANGE = (20, 24)
COMP_FONT_SIZE_RANGE = (25, 30)

PIN_TEXT_ATTEMPTS = 70
COMP_TEXT_ATTEMPTS = 100
COMP_TEXT_PROBABILITY = 0.6

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
    "v": 0.5,
    "gnd": 0.5,
    "buzzer": 0.5,
    "crystal_2pin": 0.6,
    "amp_3pin": 0.8,
    "l": 0.2,
    "c": 0.8,
    "block": 0.4,
    "ant": 0.8,
    "esd": 0.8,
    "crystal_4pin": 0.6,
    "transfomer": 1.5, # 拼写错误变体
    "jumper": 0.6,
    "amp_5pin": 0.8,
    "battary": 0.5,
    "bjt": 0.8,
    "fuse": 0.3,
    "d": 0.6,
    "bjt_npn": 0.8,
    "circle_header": 0.6,
    "dc": 0.6,
    "bjt_pnp": 0.8,
    "crystal_3pin": 0.6,
}

Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]
PinId = Tuple[str, str]
Edge = Tuple[Point, Point]


# ===================== 基础几何 =====================

def expand_rect(rect, margin):
    x0, y0, x1, y1 = rect
    return (
        x0 - margin,
        y0 - margin,
        x1 + margin,
        y1 + margin,
    )


def rects_overlap(a, b):
    return not (
        a[2] < b[0]
        or b[2] < a[0]
        or a[3] < b[1]
        or b[3] < a[1]
    )


def strictly_inside(point, rect):
    x, y = point
    x0, y0, x1, y1 = rect
    return x0 < x < x1 and y0 < y < y1


def in_board(point):
    return (
        0 <= point[0] < BOARD_W
        and 0 <= point[1] < BOARD_H
    )


def canonical_edge(a, b):
    return (a, b) if a <= b else (b, a)


def segment_hits_body(a, b, rect):
    """判断正交线段中心线是否穿入主体内部。"""
    x0, y0, x1, y1 = rect

    if a[1] == b[1]:
        y = a[1]
        lo, hi = sorted((a[0], b[0]))
        return (
            y0 < y < y1
            and max(lo, x0) < min(hi, x1)
        )

    if a[0] == b[0]:
        x = a[0]
        lo, hi = sorted((a[1], b[1]))
        return (
            x0 < x < x1
            and max(lo, y0) < min(hi, y1)
        )

    raise ValueError(f"非正交线段: {a} -> {b}")


def segment_pixels(a, b):
    """返回包含端点的像素切片。"""
    if a[1] == b[1]:
        lo, hi = sorted((a[0], b[0]))
        return a[1], slice(lo, hi + 1)

    if a[0] == b[0]:
        lo, hi = sorted((a[1], b[1]))
        return slice(lo, hi + 1), a[0]

    raise ValueError(f"非正交线段: {a} -> {b}")


def segment_intersection(a, b, c, d):
    """
    正交线段交集。

    返回：
        None：不相交
        ("point", point)：只在一个点相交
        ("overlap", None)：有非零长度的共线重叠
    """
    horizontal_ab = a[1] == b[1]
    horizontal_cd = c[1] == d[1]

    if horizontal_ab and horizontal_cd:
        if a[1] != c[1]:
            return None

        lo = max(min(a[0], b[0]), min(c[0], d[0]))
        hi = min(max(a[0], b[0]), max(c[0], d[0]))

        if lo > hi:
            return None
        if lo == hi:
            return "point", (lo, a[1])
        return "overlap", None

    if not horizontal_ab and not horizontal_cd:
        if a[0] != c[0]:
            return None

        lo = max(min(a[1], b[1]), min(c[1], d[1]))
        hi = min(max(a[1], b[1]), max(c[1], d[1]))

        if lo > hi:
            return None
        if lo == hi:
            return "point", (a[0], lo)
        return "overlap", None

    if horizontal_ab:
        ha, hb, va, vb = a, b, c, d
    else:
        ha, hb, va, vb = c, d, a, b

    point = (va[0], ha[1])

    if (
        min(ha[0], hb[0]) <= point[0] <= max(ha[0], hb[0])
        and min(va[1], vb[1]) <= point[1] <= max(va[1], vb[1])
    ):
        return "point", point

    return None


# ===================== 数据加载 =====================

def scan_component_pngs(base_path):
    result = {}

    if not base_path.exists():
        raise FileNotFoundError(base_path)

    for directory in sorted(base_path.iterdir()):
        if not directory.is_dir():
            continue

        files = [
            p
            for p in sorted(directory.glob("*.png"))
            if p.is_file() and "vis.png" not in p.name
        ]

        if files:
            result[directory.name] = files

    return result


def load_pins_info(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def collect_candidates(type_to_pngs, pins_dict):
    candidates = []
    seen = set()

    for comp_type, paths in type_to_pngs.items():
        for path in paths:
            name = path.stem
            info = pins_dict.get(name, {})

            if name in seen or not info.get("pins"):
                continue

            valid = all(
                "relative" in pin
                and "x" in pin["relative"]
                and "y" in pin["relative"]
                for pin in info["pins"].values()
            )

            if not valid or "comp_bbox" not in info:
                continue

            seen.add(name)

            candidates.append({
                "comp_name": name,
                "comp_type": comp_type,
                "png_path": path,
            })

    return candidates


def blacken(image):
    array = np.array(image.convert("RGBA")).copy()
    array[:, :, :3] = 0
    return Image.fromarray(array)


# ===================== 缩放与旋转 =====================

def rotate_coordinate(x, y, width, height, angle):
    """与 Pillow 的逆时针 transpose 旋转保持一致。"""
    if angle == 0:
        return x, y
    if angle == 90:
        return y, width - 1 - x
    if angle == 180:
        return width - 1 - x, height - 1 - y
    if angle == 270:
        return height - 1 - y, x

    raise ValueError(angle)


def prepare_component(candidate, pins_dict):
    name = candidate["comp_name"]
    info = pins_dict[name]

    with Image.open(candidate["png_path"]) as source:
        image = source.convert("RGBA")

    orig_w, orig_h = image.size
    ratio = COMPONENT_HEIGHT_RATIO.get(candidate["comp_type"], 0.7)

    target_h = max(
        2,
        round(BASE_H * ratio * random.uniform(*HEIGHT_JITTER)),
    )
    target_w = max(2, round(target_h * orig_w / orig_h))

    sx = target_w / orig_w
    sy = target_h / orig_h

    image = image.resize(
        (target_w, target_h),
        Image.Resampling.NEAREST,
    )
    image = blacken(image)

    angle = random.choices(
        ROTATION_CHOICES,
        weights=ROTATION_WEIGHTS,
        k=1,
    )[0]

    transpose = {
        90: Image.Transpose.ROTATE_90,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_270,
    }

    if angle:
        image = image.transpose(transpose[angle])

    def transform(x, y):
        rx, ry = rotate_coordinate(
            float(x) * sx,
            float(y) * sy,
            target_w,
            target_h,
            angle,
        )
        return int(round(rx)), int(round(ry))

    local_pins = {}

    for key, pin in info["pins"].items():
        relative = pin["relative"]

        local_pins[key] = {
            "point": transform(relative["x"], relative["y"]),
            "pinname": str(pin.get("pinname", key)),
        }

    raw = info["comp_bbox"]
    corners = [
        transform(raw["x1"], raw["y1"]),
        transform(raw["x1"], raw["y2"]),
        transform(raw["x2"], raw["y1"]),
        transform(raw["x2"], raw["y2"]),
    ]

    body_bbox = (
        min(p[0] for p in corners),
        min(p[1] for p in corners),
        max(p[0] for p in corners),
        max(p[1] for p in corners),
    )

    width, height = image.size

    points = (
        [(0, 0), (width - 1, height - 1)]
        + [pin["point"] for pin in local_pins.values()]
        + corners
    )

    extent = (
        min(p[0] for p in points),
        min(p[1] for p in points),
        max(p[0] for p in points),
        max(p[1] for p in points),
    )

    return {
        **candidate,
        "image": image,
        "rotation": angle,
        "scale_xy": (sx, sy),
        "local_pins": local_pins,
        "local_body_bbox": body_bbox,
        "local_extent": extent,
    }


# ===================== 器件布局 =====================

def translate_rect(rect, x, y):
    return (
        rect[0] + x,
        rect[1] + y,
        rect[2] + x,
        rect[3] + y,
    )


def random_place_components(candidates, pins_dict):
    selected = random.sample(candidates, SAMPLE_COUNT)
    prepared = [prepare_component(c, pins_dict) for c in selected]

    prepared.sort(
        key=lambda c: c["image"].width * c["image"].height,
        reverse=True,
    )

    placed = []
    step = ROUTE_GRID * 2

    # 在画布边缘给延长线额外预留空间。
    margin = max(
        BOARD_MARGIN,
        PIN_EXTENSION_LENGTH + ROUTE_GRID + WIRE_WIDTH,
    )

    for comp in prepared:
        ex0, ey0, ex1, ey1 = comp["local_extent"]

        low_x = math.ceil((margin - ex0) / step) * step
        low_y = math.ceil((margin - ey0) / step) * step

        high_x = ((BOARD_W - 1 - margin - ex1) // step) * step
        high_y = ((BOARD_H - 1 - margin - ey1) // step) * step

        if low_x > high_x or low_y > high_y:
            return None

        success = False

        for _ in range(MAX_COMPONENT_PLACE_TRIES):
            x = random.randrange(low_x, high_x + 1, step)
            y = random.randrange(low_y, high_y + 1, step)

            extent = translate_rect(comp["local_extent"], x, y)

            if any(
                rects_overlap(
                    expand_rect(extent, COMPONENT_MARGIN),
                    old["extent"],
                )
                for old in placed
            ):
                continue

            width, height = comp["image"].size

            placed.append({
                **comp,
                "origin": (x, y),
                "extent": extent,
                "bbox": (x, y, x + width - 1, y + height - 1),
                "body_bbox": translate_rect(
                    comp["local_body_bbox"], x, y
                ),
            })

            success = True
            break

        if not success:
            return None

    return placed


# ===================== Pin 向外延伸 =====================

def infer_outward_direction(point, body_bbox):
    """
    推断背向器件的正交方向。

    bbox 外部：
        选择 pin 超出主体的方向；
        角落情况下选择超出距离更大的方向。

    bbox 边界：
        选择最近的一侧。

    严格位于主体内部：
        返回 None，拒绝当前布局。
    """
    px, py = point
    x0, y0, x1, y1 = body_bbox

    if strictly_inside(point, body_bbox):
        return None

    outside = []

    if px < x0:
        outside.append((x0 - px, (-1, 0)))
    if px > x1:
        outside.append((px - x1, (1, 0)))
    if py < y0:
        outside.append((y0 - py, (0, -1)))
    if py > y1:
        outside.append((py - y1, (0, 1)))

    if outside:
        return max(outside, key=lambda item: item[0])[1]

    distances = [
        (abs(px - x0), (-1, 0)),
        (abs(px - x1), (1, 0)),
        (abs(py - y0), (0, -1)),
        (abs(py - y1), (0, 1)),
    ]

    return min(distances, key=lambda item: item[0])[1]


def make_pin_extension(point, direction):
    """
    只改变伸出方向上的坐标，保留另一坐标，
    保证 pin -> route_start 是一条直线。
    """
    x, y = point
    dx, dy = direction

    if dx < 0:
        end_x = (
            math.floor((x - PIN_EXTENSION_LENGTH) / ROUTE_GRID)
            * ROUTE_GRID
        )
        return end_x, y

    if dx > 0:
        end_x = (
            math.ceil((x + PIN_EXTENSION_LENGTH) / ROUTE_GRID)
            * ROUTE_GRID
        )
        return end_x, y

    if dy < 0:
        end_y = (
            math.floor((y - PIN_EXTENSION_LENGTH) / ROUTE_GRID)
            * ROUTE_GRID
        )
        return x, end_y

    end_y = (
        math.ceil((y + PIN_EXTENSION_LENGTH) / ROUTE_GRID)
        * ROUTE_GRID
    )
    return x, end_y


def get_pin_positions(components):
    pins = {}
    bodies = [comp["body_bbox"] for comp in components]
    used_points = set()

    for comp in components:
        ox, oy = comp["origin"]

        for key, info in comp["local_pins"].items():
            px, py = info["point"]
            point = px + ox, py + oy

            if not in_board(point):
                return None

            if point in used_points:
                return None

            if any(strictly_inside(point, body) for body in bodies):
                return None

            direction = infer_outward_direction(
                point, comp["body_bbox"]
            )

            if direction is None:
                return None

            route_start = make_pin_extension(point, direction)

            if not in_board(route_start):
                return None

            # 延长线不能穿过任何器件主体。
            if any(
                segment_hits_body(point, route_start, body)
                for body in bodies
            ):
                return None

            used_points.add(point)

            pins[(comp["comp_name"], key)] = {
                "comp_name": comp["comp_name"],
                "pin_key": key,
                "pinname": info["pinname"],
                "point": point,
                "route_start": route_start,
                "outward_direction": direction,
                "extension": (point, route_start),
            }

    # 固定延长线之间不能重叠、相交或过度靠近。
    # 即使最后属于同一 net，也保持每个 pin 的独立伸出段。
    clearance = np.zeros((BOARD_H, BOARD_W), dtype=bool)

    for pin in pins.values():
        a, b = pin["extension"]

        actual = route_mask([(a, b)], set())

        if np.any(actual & clearance):
            return None

        clearance |= route_mask(
            [(a, b)],
            set(),
            extra=WIRE_GAP + math.ceil(WIRE_WIDTH / 2),
        )

    return pins


# ===================== 随机生成 Net =====================

def random_build_nets(pin_positions):
    pins = list(pin_positions)
    random.shuffle(pins)
    total = len(pins)

    possible = [False] * (total + 1)
    possible[0] = True

    for n in range(1, total + 1):
        possible[n] = any(
            n >= size and possible[n - size]
            for size in range(MIN_NET_SIZE, MAX_NET_SIZE + 1)
        )

    if not possible[total]:
        return None

    nets = []
    cursor = 0

    while cursor < total:
        remaining = total - cursor

        choices = [
            size
            for size in range(MIN_NET_SIZE, MAX_NET_SIZE + 1)
            if size <= remaining and possible[remaining - size]
        ]

        size = random.choice(choices)
        nets.append(pins[cursor:cursor + size])
        cursor += size

    random.shuffle(nets)
    return nets


# ===================== 非均匀网格 =====================

class RoutingGrid:
    def __init__(self, pins, bodies):
        xs = set(range(0, BOARD_W, ROUTE_GRID))
        ys = set(range(0, BOARD_H, ROUTE_GRID))

        xs.add(BOARD_W - 1)
        ys.add(BOARD_H - 1)

        self.extensions = []

        for pin in pins.values():
            for x, y in (pin["point"], pin["route_start"]):
                xs.add(x)
                ys.add(y)

            self.extensions.append(pin["extension"])

        offset = max(
            WIRE_WIDTH + WIRE_GAP,
            ROUTE_GRID // 2,
        )

        for x0, y0, x1, y1 in bodies:
            xs.update((x0 - offset, x1 + offset))
            ys.update((y0 - offset, y1 + offset))

        self.xs = sorted(x for x in xs if 0 <= x < BOARD_W)
        self.ys = sorted(y for y in ys if 0 <= y < BOARD_H)

        self.x_index = {x: i for i, x in enumerate(self.xs)}
        self.y_index = {y: i for i, y in enumerate(self.ys)}

        self.bodies = bodies
        self.body_cache = {}
        self.extension_cache = {}

    def neighbors(self, point):
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

    def body_blocked(self, a, b):
        key = canonical_edge(a, b)

        if key not in self.body_cache:
            self.body_cache[key] = any(
                segment_hits_body(a, b, body)
                for body in self.bodies
            )

        return self.body_cache[key]

    def extension_blocked(self, a, b):
        """
        A* 只能接触延长线末端。
        不能穿过、重走延长线，或接到原始 pin 和延长线中间。
        """
        key = canonical_edge(a, b)

        if key not in self.extension_cache:
            blocked = False

            for pin_point, route_start in self.extensions:
                intersection = segment_intersection(
                    a, b, pin_point, route_start
                )

                if intersection is None:
                    continue

                kind, point = intersection

                if kind == "overlap" or point != route_start:
                    blocked = True
                    break

            self.extension_cache[key] = blocked

        return self.extension_cache[key]

    def split_segment(self, a, b):
        """
        把固定延长线拆成相邻网格点之间的原始边。
        这样它与 A* 路径可以在同一张图中做拓扑处理。
        """
        if a[1] == b[1]:
            lo, hi = sorted((a[0], b[0]))
            begin = bisect.bisect_left(self.xs, lo)
            end = bisect.bisect_right(self.xs, hi)
            points = [(x, a[1]) for x in self.xs[begin:end]]

        elif a[0] == b[0]:
            lo, hi = sorted((a[1], b[1]))
            begin = bisect.bisect_left(self.ys, lo)
            end = bisect.bisect_right(self.ys, hi)
            points = [(a[0], y) for y in self.ys[begin:end]]

        else:
            raise ValueError(f"非正交延长线: {a} -> {b}")

        return {
            canonical_edge(p, q)
            for p, q in zip(points, points[1:])
        }


# ===================== A* 障碍掩码 =====================

def make_foreign_pin_mask(pins, current_net):
    """
    其他 net 的整个固定延长线提前作为障碍，
    防止前面的 net 占用尚未布线的 pin 延长区域。
    """
    image = Image.new("L", (BOARD_W, BOARD_H), 0)
    draw = ImageDraw.Draw(image)

    radius = WIRE_WIDTH + WIRE_GAP

    for key, pin in pins.items():
        if key in current_net:
            continue

        a, b = pin["extension"]

        draw.line(
            [a, b],
            fill=255,
            width=2 * radius + 1,
        )

        for x, y in (a, b):
            draw.ellipse(
                (
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                ),
                fill=255,
            )

    return np.array(image) > 0


# ===================== A* 正交布线 =====================

def astar_route(
    start,
    goals,
    grid,
    blocked,
    initial_axis=0,
):
    if blocked[start[1], start[0]]:
        return None

    goals = {
        point
        for point in goals
        if not blocked[point[1], point[0]]
    }

    if not goals:
        return None

    if start in goals:
        return [start]

    gx0 = min(p[0] for p in goals)
    gy0 = min(p[1] for p in goals)
    gx1 = max(p[0] for p in goals)
    gy1 = max(p[1] for p in goals)

    def heuristic(point):
        x, y = point
        return (
            max(gx0 - x, 0, x - gx1)
            + max(gy0 - y, 0, y - gy1)
        )

    # initial_axis 延续 pin 固定伸出段方向，
    # 在末端继续直行比立即转弯代价更低。
    initial = (start[0], start[1], initial_axis)
    costs = {initial: 0}
    parents = {}

    queue = [(heuristic(start), 0, initial)]
    expanded = 0

    while queue:
        _, cost, state = heapq.heappop(queue)

        if cost != costs.get(state):
            continue

        point = state[0], state[1]
        previous_axis = state[2]

        if point in goals:
            path = []

            while True:
                path.append((state[0], state[1]))

                if state == initial:
                    return list(reversed(path))

                state = parents[state]

        expanded += 1

        if expanded > MAX_ASTAR_STATES:
            return None

        for neighbor, axis in grid.neighbors(point):
            if grid.body_blocked(point, neighbor):
                continue

            if grid.extension_blocked(point, neighbor):
                continue

            if blocked[segment_pixels(point, neighbor)].any():
                continue

            length = (
                abs(point[0] - neighbor[0])
                + abs(point[1] - neighbor[1])
            )

            bend = (
                BEND_PENALTY
                if previous_axis and previous_axis != axis
                else 0
            )

            next_cost = cost + length + bend
            next_state = neighbor[0], neighbor[1], axis

            if next_cost >= costs.get(next_state, float("inf")):
                continue

            costs[next_state] = next_cost
            parents[next_state] = state

            heapq.heappush(
                queue,
                (
                    next_cost + heuristic(neighbor),
                    next_cost,
                    next_state,
                ),
            )

    return None


# ===================== 路由拓扑与共线合并 =====================

def graph_from_edges(edges):
    graph = defaultdict(set)

    for a, b in edges:
        if a == b:
            continue

        graph[a].add(b)
        graph[b].add(a)

    return graph


def extract_segments(edges, pin_points):
    """
    将相邻原始边合并成较长线段。

    以下位置必须保留：
        1. 原始 pin；
        2. 折点；
        3. 分支节点；
        4. 线路端点。

    普通 route_start 如果是度数为 2 的直线中间点，
    会被自动删除，从而合并延长线与后续直行线。
    """
    graph = graph_from_edges(edges)
    stops = set(pin_points)

    for point, neighbors in graph.items():
        if len(neighbors) != 2:
            stops.add(point)
            continue

        a, b = tuple(neighbors)

        is_straight = (
            a[0] == point[0] == b[0]
            or a[1] == point[1] == b[1]
        )

        if not is_straight:
            stops.add(point)

    visited = set()
    segments = []

    for start in sorted(stops):
        for neighbor in sorted(graph.get(start, ())):
            edge = canonical_edge(start, neighbor)

            if edge in visited:
                continue

            visited.add(edge)
            previous = start
            current = neighbor

            while current not in stops:
                next_point = next(
                    p for p in graph[current]
                    if p != previous
                )

                visited.add(canonical_edge(current, next_point))
                previous, current = current, next_point

            segments.append(canonical_edge(start, current))

    junctions = {
        point
        for point, neighbors in graph.items()
        if len(neighbors) >= 3
    }

    return sorted(set(segments)), junctions


def route_mask(segments, junctions, extra=0):
    image = Image.new("L", (BOARD_W, BOARD_H), 0)
    draw = ImageDraw.Draw(image)

    width = WIRE_WIDTH + 2 * extra

    for a, b in segments:
        draw.line([a, b], fill=255, width=width)

    for x, y in junctions:
        radius = JUNCTION_RADIUS + extra

        draw.ellipse(
            (
                x - radius,
                y - radius,
                x + radius,
                y + radius,
            ),
            fill=255,
        )

    return np.array(image) > 0


def mask_hits_bodies(mask, bodies):
    """按实际线宽和圆点尺寸检查主体内部。"""
    for x0, y0, x1, y1 in bodies:
        left = max(0, x0 + 1)
        top = max(0, y0 + 1)
        right = min(BOARD_W, x1)
        bottom = min(BOARD_H, y1)

        if left < right and top < bottom:
            if mask[top:bottom, left:right].any():
                return True

    return False


def graph_is_connected(edges, pin_points):
    graph = graph_from_edges(edges)

    if not pin_points or not pin_points <= set(graph):
        return False

    start = next(iter(pin_points))
    seen = {start}
    stack = [start]

    while stack:
        current = stack.pop()

        for neighbor in graph[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)

    return pin_points <= seen and set(graph) <= seen


# ===================== Net 树状布线 =====================

def route_one_net(net, pins, grid, occupied):
    if len(net) < 2:
        return None

    root, second = max(
        (
            (a, b)
            for i, a in enumerate(net)
            for b in net[i + 1:]
        ),
        key=lambda pair: (
            abs(
                pins[pair[0]]["route_start"][0]
                - pins[pair[1]]["route_start"][0]
            )
            + abs(
                pins[pair[0]]["route_start"][1]
                - pins[pair[1]]["route_start"][1]
            )
        ),
    )

    remaining = [
        pin_id
        for pin_id in net
        if pin_id not in (root, second)
    ]
    random.shuffle(remaining)
    order = [second] + remaining

    # 树的连接目标只包含路由区域。
    # 固定延长线中间不加入目标，避免分支从伸出段提前接入。
    tree_points = {pins[root]["route_start"]}

    foreign_pins = make_foreign_pin_mask(pins, set(net))
    blocked = occupied | foreign_pins
    edges = set()

    # 所有 pin 都加入完整固定延长线。
    for pin_id in net:
        a, b = pins[pin_id]["extension"]

        if blocked[segment_pixels(a, b)].any():
            return None

        edges.update(grid.split_segment(a, b))

    for pin_id in order:
        pin = pins[pin_id]
        start = pin["route_start"]
        dx, _ = pin["outward_direction"]

        initial_axis = 1 if dx else 2

        path = astar_route(
            start=start,
            goals=tree_points,
            grid=grid,
            blocked=blocked,
            initial_axis=initial_axis,
        )

        if path is None:
            return None

        edges.update(
            canonical_edge(a, b)
            for a, b in zip(path, path[1:])
        )
        tree_points.update(path)

    pin_points = {pins[pin_id]["point"] for pin_id in net}

    if not graph_is_connected(edges, pin_points):
        return None

    # 原始 pin 必须是线路端点，确保只沿指定方向伸出。
    graph = graph_from_edges(edges)

    if any(len(graph[point]) != 1 for point in pin_points):
        return None

    segments, junctions = extract_segments(edges, pin_points)

    if not segments:
        return None

    actual_mask = route_mask(segments, junctions)

    if np.any(actual_mask & occupied):
        return None

    if np.any(actual_mask & foreign_pins):
        return None

    if mask_hits_bodies(actual_mask, grid.bodies):
        return None

    return {
        "net": net,
        "segments": segments,
        "junctions": junctions,
        # 输出标注时再次从原始拓扑做共线合并。
        "raw_edges": edges,
    }


def route_all_nets(nets, pins, components):
    bodies = [comp["body_bbox"] for comp in components]
    grid = RoutingGrid(pins, bodies)

    occupied = np.zeros((BOARD_H, BOARD_W), dtype=bool)
    routes = []

    for net in sorted(nets, key=len, reverse=True):
        result = route_one_net(net, pins, grid, occupied)

        if result is None:
            return None

        routes.append(result)

        occupied |= route_mask(
            result["segments"],
            result["junctions"],
            extra=WIRE_GAP + math.ceil(WIRE_WIDTH / 2),
        )

    return routes


# ===================== 文字增强 =====================

def load_font(size):
    candidates = [
        FONT_PATH,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]

    for path in candidates:
        if path and Path(path).is_file():
            return ImageFont.truetype(str(path), size)

    return ImageFont.load_default()


def randomize_pinname(name):
    def replace(match):
        value = match.group()

        if value.isdigit():
            return "".join(
                random.choice(string.digits)
                for _ in value
            )

        chars = (
            string.ascii_uppercase
            if value.isupper()
            else string.ascii_lowercase
        )

        return "".join(random.choice(chars) for _ in value)

    return re.sub(r"[A-Za-z]+|[0-9]+", replace, str(name))


def random_component_text():
    prefix = "".join(
        random.choices(
            string.ascii_uppercase,
            k=random.randint(1, 3),
        )
    )
    number = str(random.randint(0, 9999))

    suffix = random.choice([
        "",
        "+",
        "-",
        "/",
        "%",
        "=",
        "." + str(random.randint(0, 99)),
        "V",
        "mA",
        "k",
        "Hz",
        "/NC",
    ])

    return prefix + number + suffix


def add_texts(board, components, pins):
    draw = ImageDraw.Draw(board)
    pixels = np.asarray(board.convert("RGB"))
    occupied = np.any(pixels < 245, axis=2).copy()

    records = []
    used_names = set()

    for comp in components:
        x0, y0, x1, y1 = comp["body_bbox"]

        occupied[
            max(0, y0):min(BOARD_H, y1 + 1),
            max(0, x0):min(BOARD_W, x1 + 1),
        ] = True

    def unique_name(producer):
        base = producer()
        value = base
        counter = 1

        while value in used_names:
            value = f"{base}{counter}"
            counter += 1

        used_names.add(value)
        return value

    def place_text(text, reference, attempts, is_pin):
        size = random.randint(
            *(PIN_FONT_SIZE_RANGE if is_pin else COMP_FONT_SIZE_RANGE)
        )
        font = load_font(size)

        left, top, right, bottom = draw.textbbox(
            (0, 0), text, font=font
        )

        width = right - left
        height = bottom - top

        if width <= 0 or height <= 0:
            return False

        for _ in range(attempts):
            if is_pin:
                px, py = reference
                distance = random.randint(0, 20)
                side = random.randrange(4)

                if side == 0:
                    x = px - width - distance
                    y = py - height // 2 + random.randint(-18, 18)
                elif side == 1:
                    x = px + distance
                    y = py - height // 2 + random.randint(-18, 18)
                elif side == 2:
                    x = px - width // 2 + random.randint(-18, 18)
                    y = py - height - distance
                else:
                    x = px - width // 2 + random.randint(-18, 18)
                    y = py + distance

            else:
                x0, y0, x1, y1 = reference
                distance = random.randint(0, 30)
                side = random.randrange(4)

                if side == 0:
                    x = x0 - width - distance
                    y = random.randint(y0 - height, y1)
                elif side == 1:
                    x = x1 + distance
                    y = random.randint(y0 - height, y1)
                elif side == 2:
                    x = random.randint(x0 - width, x1)
                    y = y0 - height - distance
                else:
                    x = random.randint(x0 - width, x1)
                    y = y1 + distance

            x = int(x)
            y = int(y)
            margin = 4

            bx0 = x - margin
            by0 = y - margin
            bx1 = x + width + margin
            by1 = y + height + margin

            if not (
                0 <= bx0 < bx1 <= BOARD_W
                and 0 <= by0 < by1 <= BOARD_H
            ):
                continue

            if occupied[by0:by1, bx0:bx1].any():
                continue

            draw.text(
                (x - left, y - top),
                text,
                font=font,
                fill=(0, 0, 0),
            )

            occupied[by0:by1, bx0:bx1] = True

            records.append({
                "name": text,
                "bbox": (x, y, x + width, y + height),
            })

            return True

        return False

    for pin in pins.values():
        name = unique_name(
            lambda p=pin: randomize_pinname(p["pinname"])
        )

        place_text(
            name,
            pin["point"],
            PIN_TEXT_ATTEMPTS,
            is_pin=True,
        )

    for comp in components:
        if random.random() >= COMP_TEXT_PROBABILITY:
            continue

        name = unique_name(random_component_text)

        place_text(
            name,
            comp["body_bbox"],
            COMP_TEXT_ATTEMPTS,
            is_pin=False,
        )

    return records


# ===================== 渲染 =====================

def render_image(components, routes, pins):
    board = Image.new("RGB", (BOARD_W, BOARD_H), "white")
    draw = ImageDraw.Draw(board)

    # segments 已包含固定延长线与后续 A* 路径。
    for route in routes:
        for a, b in route["segments"]:
            draw.line(
                [a, b],
                fill="black",
                width=WIRE_WIDTH,
            )

        for x, y in route["junctions"]:
            draw.ellipse(
                (
                    x - JUNCTION_RADIUS,
                    y - JUNCTION_RADIUS,
                    x + JUNCTION_RADIUS,
                    y + JUNCTION_RADIUS,
                ),
                fill="black",
            )

    for comp in components:
        board.paste(
            comp["image"],
            comp["origin"],
            comp["image"],
        )

    texts = add_texts(board, components, pins)
    return board, texts


# ===================== 标注与输出后处理 =====================

def point_to_obj_flip_y(point):
    x, y = point

    return {
        "x": float(x),
        "y": float(BOARD_H - y),
    }


def bbox_flip_y(rect):
    x0, y0, x1, y1 = rect

    return [
        float(min(x0, x1)),
        float(BOARD_H - max(y0, y1)),
        float(max(x0, x1)),
        float(BOARD_H - min(y0, y1)),
    ]


def build_annotation(components, pins, routes, texts):
    annotation = {
        "image": "board_render.png",
        "board_size": [float(BOARD_W), float(BOARD_H)],
        "coordinate_system": {
            "origin": "bottom_left",
            "y_transform": "BOARD_H - image_y",
            "bbox_format": "xmin,ymin,xmax,ymax",
        },
        "components": {},
        "pins": {},
        "nets": {},
        "text": {},
    }

    for comp in components:
        name = comp["comp_name"]

        annotation["components"][name] = {
            "type": comp["comp_type"],
            "bbox": bbox_flip_y(comp["body_bbox"]),
            "rotation": comp["rotation"],
            "scale_xy": list(comp["scale_xy"]),
        }

    for (comp_name, key), pin in pins.items():
        annotation["pins"].setdefault(comp_name, {})[key] = {
            # 原始真实 pin 坐标保持不变。
            "point": point_to_obj_flip_y(pin["point"]),
            "pinname": pin["pinname"],

            # 延长后的实际布线起点。
            "route_start": point_to_obj_flip_y(pin["route_start"]),

            # 输出坐标翻转 y，因此方向的 dy 也翻转。
            "outward_direction": [
                pin["outward_direction"][0],
                -pin["outward_direction"][1],
            ],
        }

    for index, route in enumerate(routes):
        pin_points = {
            pins[pin_id]["point"]
            for pin_id in route["net"]
        }

        # 最后输出前重新从原始边图合并共线线段。
        # 不依赖 segments 列表的排列顺序。
        merged_segments, junctions = extract_segments(
            route["raw_edges"],
            pin_points,
        )

        edges = {
            f"edge{edge_index}": [
                point_to_obj_flip_y(a),
                point_to_obj_flip_y(b),
            ]
            for edge_index, (a, b) in enumerate(
                merged_segments,
                start=1,
            )
        }

        annotation["nets"][f"net{index}"] = {
            "pins": [
                {
                    "comp_name": comp_name,
                    "pin_key": pin_key,
                }
                for comp_name, pin_key in route["net"]
            ],
            "edges": edges,
            "junctions": [
                point_to_obj_flip_y(point)
                for point in sorted(junctions)
            ],
        }

    for record in texts:
        annotation["text"][record["name"]] = bbox_flip_y(
            record["bbox"]
        )

    return annotation


# ===================== 生成入口 =====================

def generate_one_case(case_id, candidates, pins_dict):
    case_dir = OUTPUT_DIR / str(case_id)

    for attempt in range(1, MAX_LAYOUT_TRIES + 1):
        print(
            f"[case {case_id}] 尝试 {attempt}/{MAX_LAYOUT_TRIES}",
            flush=True,
        )

        components = random_place_components(
            candidates,
            pins_dict,
        )

        if components is None:
            continue

        pins = get_pin_positions(components)

        if not pins or len(pins) < 2:
            continue

        nets = random_build_nets(pins)

        if nets is None:
            continue

        routes = route_all_nets(nets, pins, components)

        if routes is None:
            continue

        board, texts = render_image(
            components,
            routes,
            pins,
        )

        annotation = build_annotation(
            components,
            pins,
            routes,
            texts,
        )

        case_dir.mkdir(parents=True, exist_ok=True)

        board.save(case_dir / "board_render.png")

        with (case_dir / "annotation.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                annotation,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"生成成功: {case_dir} | "
            f"器件={len(components)}, "
            f"pin={len(pins)}, "
            f"net={len(routes)}, "
            f"text={len(texts)}",
            flush=True,
        )

        return True

    print(
        f"[case {case_id}] 达到最大尝试次数，生成失败。"
        "可检查 pin/comp_bbox 标注，"
        "或减小 PIN_EXTENSION_LENGTH、增加器件间距。",
        flush=True,
    )

    return False


def main():
    if not (2 <= MIN_NET_SIZE <= MAX_NET_SIZE):
        raise ValueError(
            "要求 2 <= MIN_NET_SIZE <= MAX_NET_SIZE"
        )

    if ROUTE_GRID <= 0:
        raise ValueError("ROUTE_GRID 必须大于 0")

    if PIN_EXTENSION_LENGTH <= 0:
        raise ValueError(
            "PIN_EXTENSION_LENGTH 必须大于 0"
        )

    if SAMPLE_COUNT < 1:
        raise ValueError("SAMPLE_COUNT 必须大于 0")

    if len(ROTATION_CHOICES) != len(ROTATION_WEIGHTS):
        raise ValueError("旋转角度和权重数量不一致")

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    type_to_pngs = scan_component_pngs(BASE_DIR)
    pins_dict = load_pins_info(PINS_JSON_PATH)

    candidates = collect_candidates(
        type_to_pngs,
        pins_dict,
    )

    if len(candidates) < SAMPLE_COUNT:
        raise RuntimeError(
            f"有效候选器件只有 {len(candidates)} 个，"
            f"少于 SAMPLE_COUNT={SAMPLE_COUNT}。"
            "器件必须具有 pins.relative 和 comp_bbox。"
        )

    print(
        f"扫描到 {len(type_to_pngs)} 个类型，"
        f"{len(candidates)} 个有效器件",
        flush=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success = 0

    for case_id in range(NUM_CASES):
        rand_net_params()
        success += int(
            generate_one_case(
                case_id,
                candidates,
                pins_dict,
            )
        )

    print(
        f"生成完成：{success}/{NUM_CASES}",
        flush=True,
    )


if __name__ == "__main__":
    main()