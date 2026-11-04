import os
import json
import sqlite3
from PIL import Image
import imagehash

# ===================== 配置区 =====================
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1] / "data" / "Dataset200" / "200_train_cases"
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "image_db.sqlite3"
SIM_THRESHOLD = 5
# ==================================================


def init_db():
    """初始化sqlite表：phash 存储十六进制字符串，json_data存原始json文本"""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS image_index (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phash TEXT NOT NULL,
        json_data TEXT NOT NULL,
        src_folder TEXT
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_phash ON image_index(phash)")
    conn.commit()
    conn.close()


def compute_phash(img_path: str) -> imagehash.ImageHash:
    """加载图片、丢弃exif，只基于像素计算pHash"""
    with Image.open(img_path) as im:
        gray = im.convert("L")
        h = imagehash.phash(gray)
    return h


def hamming_dist(h1: imagehash.ImageHash, h2: imagehash.ImageHash) -> int:
    return h1 - h2


def build_database():
    # ✅ 【修复点】入库前先初始化表
    init_db()

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    count = 0

    for folder_name in os.listdir(ROOT_DIR):
        folder_path = os.path.join(ROOT_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        png_path = None
        json_path = None
        for f in os.listdir(folder_path):
            full_f = os.path.join(folder_path, f)
            if f.lower().endswith(".png") or f.lower().endswith(".jpg"):
                png_path = full_f
            elif f.lower().endswith(".json"):
                json_path = full_f

        if png_path is None or json_path is None:
            print(f"[WARN] {folder_name} 缺少png或者json，跳过")
            continue

        try:
            phash_obj = compute_phash(png_path)
        except Exception as e:
            print(f"[ERR] 处理 {png_path} 失败: {e}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            json_content = json.dumps(json.load(f), ensure_ascii=False)

        phash_str = str(phash_obj)
        cur.execute(
            "INSERT INTO image_index (phash, json_data, src_folder) VALUES (?, ?, ?)",
            (phash_str, json_content, folder_name)
        )
        count += 1
        print(f"[OK] {folder_name} | phash={phash_str}")

    conn.commit()
    conn.close()
    print(f"\n✅ 入库完成，一共处理 {count} 条case")


def query_image(img_path: str, threshold: int = SIM_THRESHOLD):
    if not DB_PATH.exists():
        return {}
    q_hash = compute_phash(img_path)

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("SELECT phash, json_data, src_folder FROM image_index")
    rows = cur.fetchall()
    conn.close()

    match_list = []
    for phash_str, json_text, src_folder in rows:
        db_hash = imagehash.hex_to_hash(phash_str)
        dist = hamming_dist(q_hash, db_hash)
        if dist <= threshold:
            js = json.loads(json_text)
            match_list.append((dist, js, src_folder))

    match_list.sort(key=lambda x: x[0])
    if not match_list:
        return {}
    # 返回 JSON 本体；距离/来源可通过 query_image_debug 获取
    return match_list[0][1]


def query_image_debug(img_path: str, threshold: int = SIM_THRESHOLD):
    """诊断用：返回 (distance, json, src_folder) 或 None。"""
    if not DB_PATH.exists():
        return None
    q_hash = compute_phash(img_path)
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("SELECT phash, json_data, src_folder FROM image_index").fetchall()
    conn.close()
    hits = []
    for phash_str, json_text, src_folder in rows:
        dist = hamming_dist(q_hash, imagehash.hex_to_hash(phash_str))
        if dist <= threshold:
            hits.append((dist, json.loads(json_text), src_folder))
    hits.sort(key=lambda x: x[0])
    return hits[0] if hits else None


def clear_db():
    """清空数据库（方便重新构建）"""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("🗑️ 数据库已清空")


if __name__ == "__main__":
    clear_db()

    build_database()

    # test_img = r"/data1/chenjh/PCBNet/sadsd.jpg"
    # res = query_image(test_img)
    # print(res)
