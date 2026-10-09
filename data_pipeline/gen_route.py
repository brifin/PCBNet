import heapq
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw


# ===================== 配置区 =====================

BASE_DIR = Path("/data1/chenjh/PCBNet/data/base_comps")
PINS_JSON_PATH = BASE_DIR / "pins.json"
OUTPUT_DIR = Path("/data1/chenjh/PCBNet/data/Synthetic_Data")

NUM_CASES = 10

BOARD_W = 2000
BOARD_H = 1500

SAMPLE_COUNT = 5
BASE_H = 200
HEIGHT_JITTER = (0.85, 1.15)

# 器件布局间距
COMPONENT_MARGIN = 80

# 布线参数
ROUTE_GRID = 10
ROUTE_CLEARANCE = 20
WIRE_WIDTH = 4
JUNCTION_RADIUS = 6

# 每个 net 中 pin 的数量范围
MIN_NET_SIZE = 2
MAX_NET_SIZE = 4

MAX_LAYOUT_TRIES = 100
MAX_COMPONENT_PLACE_TRIES = 500

# 固定随机种子时，可以复现实验；None 表示每次随机
RANDOM_SEED = None

COMPONENT_HEIGHT_RATIO = {
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
}

Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]
PinId = Tuple[str, str]


# ===================== 基础函数 =====================

def snap_value(value: int, grid: int) -> int:
    return int(round(value / grid) * grid)


def snap_point(point: Point, grid: int) -> Point:
    return (
        snap_value(point[0], grid),
        snap_value(point[1], grid),
    )


def expand_rect(rect: Rect, margin: int) -> Rect:
    x0, y0, x1, y1 = rect
    return (
        x0 - margin,
        y0 - margin,
        x1 + margin,
        y1 + margin,
    )


def rects_overlap(rect1: Rect, rect2: Rect) -> bool:
    ax0, ay0, ax1, ay1 = rect1
    bx0, by0, bx1, by1 = rect2

    return not (
        ax1 < bx0
        or ax0 > bx1
        or ay1 < by0
        or ay0 > by1
    )


def point_in_rect(point: Point, rect: Rect) -> bool:
    x, y = point
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def scan_component_pngs(base_path: Path) -> Dict[str, List[Path]]:
    type_to_pngs: Dict[str, List[Path]] = {}

    if not base_path.exists():
        raise FileNotFoundError(f"器件目录不存在: {base_path}")

    for type_dir in sorted(base_path.iterdir()):
        if not type_dir.is_dir():
            continue

        pngs = sorted(
            path
            for path in type_dir.glob("*.png")
            if path.is_file() and '_vis.png' not in path.name
        )

        if pngs:
            type_to_pngs[type_dir.name] = pngs

    return type_to_pngs


def load_pins_info(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(f"pins.json 不存在: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def set_non_transparent_color(
    image: Image.Image,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    image = image.convert("RGBA")
    array = np.array(image).copy()

    visible = array[:, :, 3] > 0
    array[visible, 0] = color[0]
    array[visible, 1] = color[1]
    array[visible, 2] = color[2]

    return Image.fromarray(array, mode="RGBA")


# ===================== 器件选择和布局 =====================

def collect_candidates(
    type_to_pngs: Dict[str, List[Path]],
    pins_dict: dict,
) -> List[dict]:
    candidates = []

    for comp_type, png_paths in type_to_pngs.items():
        for png_path in png_paths:
            comp_name = png_path.stem
            comp_info = pins_dict.get(comp_name)

            if not comp_info:
                continue

            if not comp_info.get("pins"):
                continue

            candidates.append(
                {
                    "comp_name": comp_name,
                    "comp_type": comp_type,
                    "png_path": png_path,
                }
            )

    return candidates


def get_scaled_pin_offsets(
    comp_name: str,
    pins_dict: dict,
    scale: float,
) -> List[Point]:
    offsets = []

    for pin_info in pins_dict[comp_name].get("pins", {}).values():
        relative = pin_info.get("relative", {})
        rel_x = float(relative.get("x", 0.0))
        rel_y = float(relative.get("y", 0.0))

        offsets.append(
            (
                int(round(rel_x * scale)),
                int(round(rel_y * scale)),
            )
        )

    return offsets


def prepare_component(
    candidate: dict,
    pins_dict: dict,
) -> dict:
    image = Image.open(candidate["png_path"]).convert("RGBA")
    orig_w, orig_h = image.size

    ratio = COMPONENT_HEIGHT_RATIO.get(
        candidate["comp_type"],
        0.7,
    )

    target_h = max(
        2,
        int(
            round(
                BASE_H
                * ratio
                * random.uniform(*HEIGHT_JITTER)
            )
        ),
    )

    target_w = max(
        2,
        int(round(target_h * orig_w / orig_h)),
    )

    scale = target_h / orig_h

    resized = image.resize(
        (target_w, target_h),
        Image.Resampling.NEAREST,
    )
    resized = set_non_transparent_color(resized)

    pin_offsets = get_scaled_pin_offsets(
        candidate["comp_name"],
        pins_dict,
        scale,
    )

    return {
        **candidate,
        "image": resized,
        "orig_size": (orig_w, orig_h),
        "size": (target_w, target_h),
        "scale": scale,
        "pin_offsets": pin_offsets,
    }


def random_place_components(
    candidates: List[dict],
    pins_dict: dict,
    sample_count: int,
) -> Optional[List[dict]]:
    if len(candidates) < sample_count:
        raise RuntimeError(
            f"只有 {len(candidates)} 个带 pin 的可用器件，"
            f"无法选择 {sample_count} 个"
        )

    selected = random.sample(candidates, sample_count)
    prepared = [
        prepare_component(candidate, pins_dict)
        for candidate in selected
    ]

    # 大器件优先放置，降低布局失败概率
    prepared.sort(
        key=lambda item: item["size"][0] * item["size"][1],
        reverse=True,
    )

    placed: List[dict] = []
    layout_step = ROUTE_GRID * 4

    for component in prepared:
        target_w, target_h = component["size"]
        offsets = component["pin_offsets"]

        # 把器件和伸出 PNG 外部的 pin 都限制在画布内
        min_offset_x = min([0] + [p[0] for p in offsets])
        min_offset_y = min([0] + [p[1] for p in offsets])
        max_offset_x = max([target_w] + [p[0] for p in offsets])
        max_offset_y = max([target_h] + [p[1] for p in offsets])

        min_x = max(
            COMPONENT_MARGIN,
            COMPONENT_MARGIN - min_offset_x,
        )
        min_y = max(
            COMPONENT_MARGIN,
            COMPONENT_MARGIN - min_offset_y,
        )
        max_x = min(
            BOARD_W - target_w - COMPONENT_MARGIN,
            BOARD_W - max_offset_x - COMPONENT_MARGIN,
        )
        max_y = min(
            BOARD_H - target_h - COMPONENT_MARGIN,
            BOARD_H - max_offset_y - COMPONENT_MARGIN,
        )

        min_x = (
            (min_x + layout_step - 1)
            // layout_step
            * layout_step
        )
        min_y = (
            (min_y + layout_step - 1)
            // layout_step
            * layout_step
        )

        max_x = max_x // layout_step * layout_step
        max_y = max_y // layout_step * layout_step

        if min_x > max_x or min_y > max_y:
            return None

        placed_successfully = False

        for _ in range(MAX_COMPONENT_PLACE_TRIES):
            x0 = random.randrange(
                min_x,
                max_x + 1,
                layout_step,
            )
            y0 = random.randrange(
                min_y,
                max_y + 1,
                layout_step,
            )

            x1 = x0 + target_w - 1
            y1 = y0 + target_h - 1
            bbox = (x0, y0, x1, y1)

            collision = any(
                rects_overlap(
                    expand_rect(bbox, COMPONENT_MARGIN),
                    expand_rect(
                        old_component["bbox"],
                        COMPONENT_MARGIN,
                    ),
                )
                for old_component in placed
            )

            if collision:
                continue

            placed.append(
                {
                    **component,
                    "bbox": bbox,
                }
            )
            placed_successfully = True
            break

        if not placed_successfully:
            return None

    return placed


# ===================== Pin 坐标和引出点 =====================

def choose_pin_side(
    point: Point,
    bbox: Rect,
) -> str:
    px, py = point
    x0, y0, x1, y1 = bbox

    if px < x0:
        return "left"
    if px > x1:
        return "right"
    if py < y0:
        return "top"
    if py > y1:
        return "bottom"

    distances = {
        "left": abs(px - x0),
        "right": abs(x1 - px),
        "top": abs(py - y0),
        "bottom": abs(y1 - py),
    }

    return min(distances, key=distances.get)


def make_pin_stub(
    pin_point: Point,
    component_bbox: Rect,
) -> Tuple[List[Point], Point]:
    """
    生成 pin 到器件外部布线网格的引出线。

    返回：
        stub_path: pin 到布线起点的正交路径
        route_start: A* 布线起点
    """
    px, py = pin_point
    obstacle = expand_rect(
        component_bbox,
        ROUTE_CLEARANCE,
    )
    x0, y0, x1, y1 = obstacle

    side = choose_pin_side(
        pin_point,
        component_bbox,
    )

    if side == "left":
        outside_x = snap_value(
            x0 - ROUTE_GRID,
            ROUTE_GRID,
        )
        route_start = (
            outside_x,
            snap_value(py, ROUTE_GRID),
        )
        stub = [
            pin_point,
            (outside_x, py),
            route_start,
        ]

    elif side == "right":
        outside_x = snap_value(
            x1 + ROUTE_GRID,
            ROUTE_GRID,
        )
        route_start = (
            outside_x,
            snap_value(py, ROUTE_GRID),
        )
        stub = [
            pin_point,
            (outside_x, py),
            route_start,
        ]

    elif side == "top":
        outside_y = snap_value(
            y0 - ROUTE_GRID,
            ROUTE_GRID,
        )
        route_start = (
            snap_value(px, ROUTE_GRID),
            outside_y,
        )
        stub = [
            pin_point,
            (px, outside_y),
            route_start,
        ]

    else:
        outside_y = snap_value(
            y1 + ROUTE_GRID,
            ROUTE_GRID,
        )
        route_start = (
            snap_value(px, ROUTE_GRID),
            outside_y,
        )
        stub = [
            pin_point,
            (px, outside_y),
            route_start,
        ]

    # 删除连续重复点
    cleaned = [stub[0]]

    for point in stub[1:]:
        if point != cleaned[-1]:
            cleaned.append(point)

    return cleaned, route_start


def get_pin_positions(
    placed_components: List[dict],
    pins_dict: dict,
) -> Dict[PinId, dict]:
    pin_positions: Dict[PinId, dict] = {}

    for component in placed_components:
        comp_name = component["comp_name"]
        x0, y0, _, _ = component["bbox"]
        scale = component["scale"]

        for pin_key, pin_info in pins_dict[
            comp_name
        ].get("pins", {}).items():
            relative = pin_info.get("relative", {})

            rel_x = float(relative.get("x", 0.0))
            rel_y = float(relative.get("y", 0.0))

            pin_point = (
                int(round(x0 + rel_x * scale)),
                int(round(y0 + rel_y * scale)),
            )

            stub, route_start = make_pin_stub(
                pin_point,
                component["bbox"],
            )

            if not (
                0 <= route_start[0] < BOARD_W
                and 0 <= route_start[1] < BOARD_H
            ):
                return {}

            pin_positions[(comp_name, pin_key)] = {
                "comp_name": comp_name,
                "pin_key": pin_key,
                "pinname": pin_info.get(
                    "pinname",
                    pin_key,
                ),
                "point": pin_point,
                "stub": stub,
                "route_start": route_start,
            }

    return pin_positions


# ===================== 随机划分 Net =====================

def random_build_nets(
    pin_positions: Dict[PinId, dict],
) -> List[List[PinId]]:
    pins = list(pin_positions.keys())
    random.shuffle(pins)

    if len(pins) < 2:
        raise RuntimeError("总 pin 数不足 2，无法生成 net")

    nets: List[List[PinId]] = []
    cursor = 0

    while cursor < len(pins):
        remaining = len(pins) - cursor

        if remaining <= MAX_NET_SIZE:
            if remaining >= MIN_NET_SIZE:
                nets.append(pins[cursor:])
                break

            nets[-1].extend(pins[cursor:])
            break

        valid_sizes = [
            size
            for size in range(
                MIN_NET_SIZE,
                MAX_NET_SIZE + 1,
            )
            if remaining - size != 1
        ]

        net_size = random.choice(valid_sizes)
        nets.append(
            pins[cursor:cursor + net_size]
        )
        cursor += net_size

    random.shuffle(nets)
    return nets


# ===================== 路径工具 =====================

def build_obstacles(
    placed_components: List[dict],
) -> List[Rect]:
    obstacles = []

    for component in placed_components:
        rect = expand_rect(
            component["bbox"],
            ROUTE_CLEARANCE,
        )

        obstacles.append(
            (
                max(0, rect[0]),
                max(0, rect[1]),
                min(BOARD_W - 1, rect[2]),
                min(BOARD_H - 1, rect[3]),
            )
        )

    return obstacles


def is_blocked(
    point: Point,
    obstacles: List[Rect],
    occupied: Set[Point],
    allowed: Set[Point],
) -> bool:
    if point in allowed:
        return False

    if point in occupied:
        return True

    return any(
        point_in_rect(point, rect)
        for rect in obstacles
    )


def simplify_path(path: List[Point]) -> List[Point]:
    if len(path) <= 2:
        return path

    simplified = [path[0]]

    for index in range(1, len(path) - 1):
        prev_point = path[index - 1]
        current = path[index]
        next_point = path[index + 1]

        first_horizontal = prev_point[1] == current[1]
        second_horizontal = current[1] == next_point[1]

        if first_horizontal != second_horizontal:
            simplified.append(current)

    simplified.append(path[-1])
    return simplified


def interpolate_path(
    path: List[Point],
    grid: int,
) -> List[Point]:
    if len(path) <= 1:
        return path[:]

    points: List[Point] = []

    for index in range(len(path) - 1):
        x0, y0 = path[index]
        x1, y1 = path[index + 1]

        if x0 == x1:
            step = grid if y1 >= y0 else -grid
            segment = [
                (x0, y)
                for y in range(
                    y0,
                    y1 + step,
                    step,
                )
            ]
        elif y0 == y1:
            step = grid if x1 >= x0 else -grid
            segment = [
                (x, y0)
                for x in range(
                    x0,
                    x1 + step,
                    step,
                )
            ]
        else:
            raise ValueError(
                f"出现非正交路径: "
                f"{path[index]} -> {path[index + 1]}"
            )

        if index > 0:
            segment = segment[1:]

        points.extend(segment)

    return points


# ===================== A* 正交布线 =====================

def astar_route(
    start: Point,
    goals: Set[Point],
    obstacles: List[Rect],
    occupied: Set[Point],
) -> Optional[List[Point]]:
    if not goals:
        return None

    allowed = set(goals)
    allowed.add(start)

    directions = {
        1: (0, -ROUTE_GRID),
        2: (0, ROUTE_GRID),
        3: (-ROUTE_GRID, 0),
        4: (ROUTE_GRID, 0),
    }

    def heuristic(point: Point) -> int:
        return min(
            abs(point[0] - goal[0])
            + abs(point[1] - goal[1])
            for goal in goals
        )

    start_state = (start[0], start[1], 0)

    queue = [
        (
            heuristic(start),
            0,
            start[0],
            start[1],
            0,
        )
    ]

    costs = {start_state: 0}
    parents = {}
    final_state = None

    while queue:
        _, current_cost, x, y, previous_direction = (
            heapq.heappop(queue)
        )

        state = (x, y, previous_direction)

        if current_cost != costs.get(state):
            continue

        if (x, y) in goals:
            final_state = state
            break

        for direction, (dx, dy) in directions.items():
            next_point = (x + dx, y + dy)

            if not (
                0 <= next_point[0] < BOARD_W
                and 0 <= next_point[1] < BOARD_H
            ):
                continue

            if is_blocked(
                next_point,
                obstacles,
                occupied,
                allowed,
            ):
                continue

            bend_cost = 0

            if (
                previous_direction != 0
                and previous_direction != direction
            ):
                # 较高的转弯代价使线路更加方正
                bend_cost = ROUTE_GRID * 12

            next_cost = (
                current_cost
                + ROUTE_GRID
                + bend_cost
            )

            next_state = (
                next_point[0],
                next_point[1],
                direction,
            )

            if next_cost >= costs.get(
                next_state,
                float("inf"),
            ):
                continue

            costs[next_state] = next_cost
            parents[next_state] = state

            heapq.heappush(
                queue,
                (
                    next_cost + heuristic(next_point),
                    next_cost,
                    next_point[0],
                    next_point[1],
                    direction,
                ),
            )

    if final_state is None:
        return None

    path = []

    while True:
        path.append(
            (
                final_state[0],
                final_state[1],
            )
        )

        if final_state == start_state:
            break

        final_state = parents[final_state]

    path.reverse()
    return simplify_path(path)


# ===================== Net 布线 =====================

def route_one_net(
    net: List[PinId],
    pin_positions: Dict[PinId, dict],
    obstacles: List[Rect],
    occupied_by_other_nets: Set[Point],
) -> Optional[dict]:
    """
    先连接两个 pin，后续 pin 接入已有线路。

    第三个及之后的 pin 与已有线路的接入位置会被记录为
    junction，并绘制实心圆。
    """
    if len(net) < 2:
        return None

    # 优先从相距较远的两个 pin 开始形成主干
    best_pair = None
    best_distance = -1

    for i in range(len(net)):
        for j in range(i + 1, len(net)):
            point1 = pin_positions[
                net[i]
            ]["route_start"]
            point2 = pin_positions[
                net[j]
            ]["route_start"]

            distance = (
                abs(point1[0] - point2[0])
                + abs(point1[1] - point2[1])
            )

            if distance > best_distance:
                best_distance = distance
                best_pair = (net[i], net[j])

    root_pin, second_pin = best_pair
    ordered_pins = [root_pin, second_pin]

    remaining = [
        pin
        for pin in net
        if pin not in {root_pin, second_pin}
    ]
    random.shuffle(remaining)
    ordered_pins.extend(remaining)

    root_start = pin_positions[
        root_pin
    ]["route_start"]

    tree_points: Set[Point] = {root_start}
    wire_points: Set[Point] = {root_start}

    paths: List[List[Point]] = []
    junctions: Set[Point] = set()

    # 每个 pin 的短引出线都需要绘制
    stubs = [
        pin_positions[pin]["stub"]
        for pin in ordered_pins
    ]

    for pin_index, pin_id in enumerate(
        ordered_pins[1:],
        start=1,
    ):
        start = pin_positions[
            pin_id
        ]["route_start"]

        path = astar_route(
            start=start,
            goals=tree_points,
            obstacles=obstacles,
            occupied=occupied_by_other_nets,
        )

        if path is None:
            return None

        full_points = interpolate_path(
            path,
            ROUTE_GRID,
        )

        if not full_points:
            return None

        connection_point = full_points[-1]

        # 第三根及后续线路接入已有主线时画实心点
        if pin_index >= 2:
            junctions.add(connection_point)

        paths.append(path)
        tree_points.update(full_points)
        wire_points.update(full_points)

    return {
        "net": net,
        "ordered_pins": ordered_pins,
        "stubs": stubs,
        "paths": paths,
        "junctions": junctions,
        "wire_points": wire_points,
    }


def route_all_nets(
    nets: List[List[PinId]],
    pin_positions: Dict[PinId, dict],
    placed_components: List[dict],
) -> Optional[List[dict]]:
    obstacles = build_obstacles(
        placed_components,
    )

    # 多 pin net 更难，优先布线
    sorted_nets = sorted(
        nets,
        key=len,
        reverse=True,
    )

    routes = []
    occupied: Set[Point] = set()

    for net in sorted_nets:
        result = route_one_net(
            net=net,
            pin_positions=pin_positions,
            obstacles=obstacles,
            occupied_by_other_nets=occupied,
        )

        if result is None:
            return None

        routes.append(result)
        occupied.update(result["wire_points"])

    return routes


# ===================== 绘制和标注 =====================

def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: List[Point],
    width: int,
) -> None:
    cleaned = []

    for point in points:
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)

    if len(cleaned) >= 2:
        draw.line(
            cleaned,
            fill=(0, 0, 0),
            width=width,
            joint="curve",
        )


def render_image(
    placed_components: List[dict],
    routes: List[dict],
) -> Image.Image:
    board = Image.new(
        "RGB",
        (BOARD_W, BOARD_H),
        "white",
    )
    draw = ImageDraw.Draw(board)

    # 先绘制导线
    for route in routes:
        for stub in route["stubs"]:
            draw_polyline(
                draw,
                stub,
                WIRE_WIDTH,
            )

        for path in route["paths"]:
            draw_polyline(
                draw,
                path,
                WIRE_WIDTH,
            )

    # 绘制联通分支的实心圆点
    for route in routes:
        for x, y in route["junctions"]:
            draw.ellipse(
                (
                    x - JUNCTION_RADIUS,
                    y - JUNCTION_RADIUS,
                    x + JUNCTION_RADIUS,
                    y + JUNCTION_RADIUS,
                ),
                fill=(0, 0, 0),
            )

    # 最后绘制器件，防止线路覆盖器件主体
    for component in placed_components:
        x0, y0, _, _ = component["bbox"]

        board.paste(
            component["image"],
            (x0, y0),
            component["image"],
        )

    return board


def point_list_to_json(
    points: List[Point],
) -> List[List[int]]:
    return [
        [int(x), int(y)]
        for x, y in points
    ]


def build_annotation(
    placed_components: List[dict],
    pin_positions: Dict[PinId, dict],
    routes: List[dict],
) -> dict:
    annotation = {
        "image": "board_render.png",
        "board_size": [BOARD_W, BOARD_H],
        "components": [],
        "nets": [],
    }

    for component in placed_components:
        annotation["components"].append(
            {
                "comp_name": component["comp_name"],
                "type": component["comp_type"],
                "png_path": str(
                    component["png_path"]
                ),
                "bbox": list(component["bbox"]),
                "scale": component["scale"],
            }
        )

    for net_id, route in enumerate(routes):
        pins = []

        for pin_id in route["net"]:
            pin = pin_positions[pin_id]

            pins.append(
                {
                    "comp_name": pin["comp_name"],
                    "pin_key": pin["pin_key"],
                    "pinname": pin["pinname"],
                    "point": list(pin["point"]),
                    "route_start": list(
                        pin["route_start"]
                    ),
                    "stub": point_list_to_json(
                        pin["stub"]
                    ),
                }
            )

        annotation["nets"].append(
            {
                "net_id": net_id,
                "pins": pins,
                "paths": [
                    point_list_to_json(path)
                    for path in route["paths"]
                ],
                "junctions": point_list_to_json(
                    sorted(route["junctions"])
                ),
            }
        )

    return annotation


# ===================== 样本生成 =====================

def generate_one_case(
    case_id: int,
    candidates: List[dict],
    pins_dict: dict,
) -> bool:
    case_dir = OUTPUT_DIR / str(case_id)

    for attempt in range(
        1,
        MAX_LAYOUT_TRIES + 1,
    ):
        print(
            f"[case {case_id}] "
            f"尝试 {attempt}/{MAX_LAYOUT_TRIES}"
        )

        placed_components = random_place_components(
            candidates=candidates,
            pins_dict=pins_dict,
            sample_count=SAMPLE_COUNT,
        )

        if placed_components is None:
            continue

        pin_positions = get_pin_positions(
            placed_components,
            pins_dict,
        )

        if len(pin_positions) < 2:
            continue

        nets = random_build_nets(
            pin_positions,
        )

        routes = route_all_nets(
            nets=nets,
            pin_positions=pin_positions,
            placed_components=placed_components,
        )

        if routes is None:
            continue

        board = render_image(
            placed_components,
            routes,
        )

        annotation = build_annotation(
            placed_components,
            pin_positions,
            routes,
        )

        case_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_path = (
            case_dir / "board_render.png"
        )
        annotation_path = (
            case_dir / "annotation.json"
        )

        board.save(image_path)

        with annotation_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                annotation,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(f"生成成功: {image_path}")
        print(
            f"器件数={len(placed_components)}, "
            f"pin数={len(pin_positions)}, "
            f"net数={len(routes)}"
        )

        return True

    print(
        f"[case {case_id}] "
        "达到最大尝试次数，生成失败"
    )
    return False


def main() -> None:
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    type_to_pngs = scan_component_pngs(
        BASE_DIR,
    )
    pins_dict = load_pins_info(
        PINS_JSON_PATH,
    )

    candidates = collect_candidates(
        type_to_pngs,
        pins_dict,
    )

    print(
        f"扫描到 {len(type_to_pngs)} 个类型，"
        f"{len(candidates)} 个带 pin 的可用器件"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    success_count = 0

    for case_id in range(NUM_CASES):
        if generate_one_case(
            case_id,
            candidates,
            pins_dict,
        ):
            success_count += 1

    print(
        f"生成完成: "
        f"{success_count}/{NUM_CASES}"
    )


if __name__ == "__main__":
    main()
