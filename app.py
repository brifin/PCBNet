#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PCB schematic annotation editor (Flask).

四种模式：
  pred    —— 对比评测结果：JSON 取 scoring_result/final_results/<id>_Pred.json，
             图片与 GT 取 10gt 的原图/标注；
  train   —— 编辑 200_train_cases（原图 + *_target 标注同目录）；
  modify  —— 专门修改 200_train_cases 的元件 bbox：单面板，点击元件/框选后
             显示并重画 bbox，或直接输入四个坐标；保存回 *_target.json；
  review  —— 检查模式：只有原图（无 GT），可直接上传图片、运行当前模型、
             修改 JSON；默认目录在 data/Dataset750/200_train_cases 存在时指向
             该目录（750 例 Dataset750），否则用 data/Dataset200/review_cases。
             可用 scripts_extra/prepare_review_cases.py 把收集的图片整理成
             review_cases/<id>/<id>.png + review_cases/<id>/<id>.json。

review 模式既可通过 APP_MODE=review 启动，也可在默认实例里点右上角“检查模式”
（/review），互不影响。modify 可通过 APP_MODE=modify 或 /modify 进入。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

MODE = os.environ.get("APP_MODE", "pred")
if MODE not in {"pred", "train", "modify", "review"}:
    MODE = "pred"

TRAIN_DIR = Path(os.environ.get("APP_TRAIN_DIR", ROOT / "data" / "Dataset200" / "200_train_cases"))
GT_IMG_DIR = ROOT / "data" / "Dataset200" / "10gt"
# PRED_DIR = Path("/data/luoxuan/PCBNet/outputs/pred")
PRED_DIR = ROOT / "scoring_result" / "final_results"


def _has_review_cases(path: Path) -> bool:
    if not path.is_dir():
        return False
    for d in path.iterdir():
        if not d.is_dir():
            continue
        if any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in d.iterdir()):
            return True
    return False


def _default_review_dir() -> Path:
    return ROOT / "data" / "Dataset"


REVIEW_DIR = Path(os.environ.get("APP_REVIEW_DIR", _default_review_dir()))

EMPTY_JSON = {"components": {}, "pins": {}, "nets": {}}
_PREDICT_LOCK = threading.Lock()

app = Flask(__name__)


# ----------------------------------------------------------------------------
# 路径 / 模式
# ----------------------------------------------------------------------------

def _mode() -> str:
    m = request.args.get("mode") or MODE
    return m if m in {"pred", "train", "modify", "review"} else MODE


def _valid_cid(cid: str) -> bool:
    return bool(cid) and cid == Path(cid).name and cid not in {".", ".."}


def _safe_new_cid(filename: str) -> str:
    stem = Path(filename or "").stem
    cid = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in stem).strip("_")
    return (cid or f"case_{int(time.time())}")[:80]


def _case_sort_key(name: str):
    return (0, int(name)) if name.isdigit() else (1, name.lower())


def _pred_json(cid: str) -> Path:
    """本地测试结果：优先 pred/<id>.json（扁平），兼容旧的 <id>/<id>.json 和平台 <id>_Pred.json。"""
    flat = PRED_DIR / f"{cid}.json"
    if flat.exists():
        return flat
    platform = PRED_DIR / f"{cid}_Pred.json"
    if platform.exists():
        return platform
    return PRED_DIR / cid / f"{cid}.json"


def _review_case_dir(cid: str, create: bool = False) -> Path | None:
    if not _valid_cid(cid):
        return None
    d = REVIEW_DIR / cid
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d if d.is_dir() else None


def _review_image(cid: str) -> Path | None:
    d = _review_case_dir(cid)
    if d is None:
        return None
    imgs = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return imgs[0] if imgs else None


def _review_pred(cid: str) -> Path:
    d = _review_case_dir(cid, create=True)
    return d / f"{cid}_Pred.json"


def _review_target(cid: str) -> Path:
    d = _review_case_dir(cid, create=True)
    return d / f"{cid}_target.json"


def _review_work_json(cid: str) -> Path | None:
    """检查模式当前编辑的 JSON：已保存标注 > <id>.json（整理后的可编辑文件）> 模型 Pred。"""
    d = _review_case_dir(cid)
    if d is None:
        return None
    for p in (d / f"{cid}_target.json", d / f"{cid}.json", d / f"{cid}_Pred.json"):
        if p.exists():
            return p
    return None


def _review_json_is_empty(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return not any((data.get(k) or {}) for k in ("components", "pins", "nets"))


def _case_dirs(mode: str | None = None):
    mode = mode or MODE
    if mode == "review":
        if not REVIEW_DIR.is_dir():
            return []
        ids = [d.name for d in REVIEW_DIR.iterdir() if d.is_dir() and _review_image(d.name)]
        return sorted(ids, key=_case_sort_key)
    if mode == "pred":
        ids = sorted(
            (d.name for d in GT_IMG_DIR.iterdir() if d.is_dir() and d.name.isdigit()),
            key=int,
        )
        return [i for i in ids if _pred_json(i).exists()]
    if not TRAIN_DIR.is_dir():
        return []
    ids = sorted((d.name for d in TRAIN_DIR.iterdir() if d.is_dir() and d.name.isdigit()), key=int)
    if mode == "modify":
        return [i for i in ids if any("_target" in p.name and p.suffix == ".json" for p in (TRAIN_DIR / i).glob("*.json"))]
    return ids


def _files(cid: str, mode: str | None = None):
    """返回 (case_dir, image_path, editable/pred json_path)。"""
    mode = mode or _mode()
    if not _valid_cid(cid):
        return None, None, None
    if mode == "review":
        d = _review_case_dir(cid)
        if d is None:
            return None, None, None
        return d, _review_image(cid), _review_work_json(cid)
    if mode == "pred":
        img = GT_IMG_DIR / cid / f"{cid}.png"
        js = _pred_json(cid)
        return GT_IMG_DIR / cid, (img if img.exists() else None), (js if js.exists() else None)
    d = TRAIN_DIR / cid
    if not d.is_dir():
        return None, None, None
    img = next((p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS), None)
    js = next((p for p in d.iterdir() if "_target" in p.name and p.suffix == ".json"), None)
    return d, img, js


def _gt_json(cid: str, mode: str | None = None):
    mode = mode or _mode()
    if mode in {"review", "modify"} or not _valid_cid(cid):
        return None
    if mode == "pred":
        p = GT_IMG_DIR / cid / f"{cid}.json"
        return p if p.exists() else None
    d = TRAIN_DIR / cid
    if not d.is_dir():
        return None
    return next((p for p in d.iterdir() if "_target" in p.name and p.suffix == ".json"), None)


def _image_size(img: Path | None):
    if img is None:
        return 0, 0
    raw = np.fromfile(str(img), dtype=np.uint8)
    im = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if im is None:
        return 0, 0
    return int(im.shape[1]), int(im.shape[0])


# ----------------------------------------------------------------------------
# JSON 规范化
# ----------------------------------------------------------------------------

def _fix_hypergraph(hg) -> str:
    """规范化 hyperGraph 为 "(ref1,ref2,...)"，修复缺失括号等脏数据。"""
    s = str(hg or "").replace("(", "").replace(")", "")
    refs = [r.strip() for r in s.split(",") if r.strip()]
    return "(" + ",".join(refs) + ")"


def _r1(x):
    """坐标数值保留一位小数（bbox/point/edge 用），非数值原样返回。"""
    try:
        return round(float(x), 1)
    except (TypeError, ValueError):
        return x


def _canonical(data: dict) -> dict:
    """按正确 JSON 字段顺序重建结构，保证写回时输出顺序一致。

    正确顺序：顶层 components → pins → nets；
    元件 Name → type → value → bbox；net hyperGraph → edges；pin pinname → point。
    未知字段保留并追加在已知字段之后。
    """
    out: dict = {"components": {}, "pins": {}, "nets": {}}

    for k, c in (data.get("components") or {}).items():
        bbox = c.get("bbox")
        nc = {"Name": c.get("Name"), "type": c.get("type"),
              "value": c.get("value"),
              "bbox": [_r1(x) for x in bbox] if isinstance(bbox, list) else bbox}
        for extra in c:
            if extra not in nc:
                nc[extra] = c[extra]
        out["components"][k] = nc

    for k, plist in (data.get("pins") or {}).items():
        npins: dict = {}
        for pk, p in (plist or {}).items():
            pt = p.get("point") or {}
            npin = {"pinname": p.get("pinname"), "point": {"x": _r1(pt.get("x")), "y": _r1(pt.get("y"))}}
            for extra in p:
                if extra not in npin:
                    npin[extra] = p[extra]
            npins[pk] = npin
        out["pins"][k] = npins

    for k, n in (data.get("nets") or {}).items():
        edges = n.get("edges") or {}
        nedges = {ek: [{"x": _r1(p.get("x")), "y": _r1(p.get("y"))} for p in e]
                  for ek, e in edges.items()}
        nn = {"hyperGraph": _fix_hypergraph(n.get("hyperGraph")), "edges": nedges}
        for extra in n:
            if extra not in nn:
                nn[extra] = n[extra]
        out["nets"][k] = nn

    for k in data:
        if k not in out:
            out[k] = data[k]
    return out


def _ensure_sections(data: dict) -> dict:
    for k in ("components", "pins", "nets"):
        if not isinstance(data.get(k), dict):
            data[k] = {}
    return data


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_canonical(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json_upload(file_storage) -> dict:
    raw = file_storage.read()
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON 顶层必须是对象")
    return data


# ----------------------------------------------------------------------------
# 页面 / 查询
# ----------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", app_mode=MODE, root_mode=MODE)


@app.route("/modify")
def modify_page():
    return render_template("index.html", app_mode="modify", root_mode=MODE)


@app.route("/review")
def review_page():
    return render_template("index.html", app_mode="review", root_mode=MODE)


@app.route("/api/cases")
def api_cases():
    return jsonify(_case_dirs(_mode()))


@app.route("/api/case/<cid>")
def api_case(cid):
    mode = _mode()
    d, img, js = _files(cid, mode)
    if mode == "review":
        if img is None:
            return jsonify({"error": f"review case {cid} not found"}), 404
        data = json.loads(js.read_text(encoding="utf-8")) if js else dict(EMPTY_JSON)
        _ensure_sections(data)
        w, h = _image_size(img)
        return jsonify({
            "case_id": cid,
            "json": data,
            "width": w,
            "height": h,
            "image": f"/image/{cid}?mode=review",
            "review": True,
            "has_pred": _review_pred(cid).exists(),
            "has_target": _review_target(cid).exists(),
            "has_annotation": _review_work_json(cid) is not None,
        })
    if js is None:
        return jsonify({"error": f"case {cid} not found"}), 404
    data = json.loads(js.read_text(encoding="utf-8"))
    _ensure_sections(data)
    w, h = _image_size(img) if img else (0, 0)
    return jsonify({"case_id": cid, "json": data, "width": w, "height": h,
                    "image": f"/image/{cid}?mode={mode}", "review": False})


@app.route("/api/gt/<cid>")
def api_gt(cid):
    gt = _gt_json(cid, _mode())
    if gt is None:
        return jsonify({"error": f"gt for case {cid} not found"}), 404
    data = json.loads(gt.read_text(encoding="utf-8"))
    _ensure_sections(data)
    return jsonify({"case_id": cid, "json": data})


@app.route("/image/<cid>")
def image(cid):
    _, img, _ = _files(cid, _mode())
    if img is None:
        return ("", 404)
    return send_file(img)


@app.route("/api/save/<cid>", methods=["POST"])
def api_save(cid):
    mode = _mode()
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"error": "JSON 顶层必须是对象"}), 400
    if mode == "review":
        if _review_case_dir(cid) is None:
            return jsonify({"error": f"review case {cid} not found"}), 404
        dest = _review_work_json(cid)
        if dest is None:
            dest = _review_case_dir(cid, create=True) / f"{cid}.json"
        _write_json(dest, data)
        return jsonify({"ok": True, "case_id": cid, "path": str(dest), "mode": mode})
    _, _, js = _files(cid, mode)
    if js is None:
        return jsonify({"error": f"case {cid} not found"}), 404
    _write_json(js, data)
    return jsonify({"ok": True, "case_id": cid, "path": str(js), "mode": mode})


# ----------------------------------------------------------------------------
# 检查模式：上传 / 模型预测 / 导入 / 加入训练集
# ----------------------------------------------------------------------------

@app.route("/api/review/upload", methods=["POST"])
def review_upload():
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify({"error": "缺少图片文件"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in IMAGE_EXTS:
        return jsonify({"error": f"不支持的图片格式：{ext}"}), 400

    # 先解析可选 Pred JSON，避免图片已落盘但 JSON 非法
    pred = None
    jf = request.files.get("pred_json")
    if jf is not None and jf.filename:
        try:
            pred = _read_json_upload(jf)
        except Exception as exc:
            return jsonify({"error": f"Pred JSON 解析失败：{exc}"}), 400

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    base = _safe_new_cid(f.filename)
    cid = base
    n = 2
    while (REVIEW_DIR / cid).exists():
        cid = f"{base}_{n}"
        n += 1
    d = REVIEW_DIR / cid
    d.mkdir(parents=True, exist_ok=True)
    img_path = d / f"{cid}{ext}"
    f.save(img_path)
    if pred is not None:
        _write_json(d / f"{cid}.json", pred)
    return jsonify({"ok": True, "case_id": cid, "image": str(img_path),
                    "has_pred": pred is not None})


@app.route("/api/review/upload_json/<cid>", methods=["POST"])
def review_upload_json(cid):
    if _review_case_dir(cid) is None:
        return jsonify({"error": f"review case {cid} not found"}), 404
    f = request.files.get("json")
    if f is None or not f.filename:
        return jsonify({"error": "缺少 JSON 文件"}), 400
    try:
        data = _read_json_upload(f)
    except Exception as exc:
        return jsonify({"error": f"JSON 解析失败：{exc}"}), 400
    target = _review_target(cid)
    overwrite = request.args.get("overwrite") == "1"
    if target.exists() and not overwrite:
        return jsonify({"error": "已存在保存的标注，是否覆盖？", "need_overwrite": True}), 409
    if target.exists():
        _write_json(target, data)
        return jsonify({"ok": True, "case_id": cid, "path": str(target), "overwritten": True})
    anno = _review_case_dir(cid, create=True) / f"{cid}.json"
    _write_json(anno, data)
    return jsonify({"ok": True, "case_id": cid, "path": str(anno), "overwritten": False})


@app.route("/api/review/predict/<cid>", methods=["POST"])
def review_predict(cid):
    mode = "review"
    d, img, _ = _files(cid, mode)
    if img is None or d is None:
        return jsonify({"error": f"review case {cid} not found"}), 404
    if not _PREDICT_LOCK.acquire(blocking=False):
        return jsonify({"error": "已有模型推理在进行，请稍后再试"}), 409
    t0 = time.time()
    try:
        # 延迟 import：只有真正点“运行模型”时才加载 torch/paddle 依赖
        from pcbnet.config import load_config
        from pcbnet.pipeline import infer_image

        cfg = load_config(ROOT / "configs" / "default.yaml")
        target = _review_target(cid)
        anno = d / f"{cid}.json"
        warning = ""
        if target.exists():
            out_path = _review_pred(cid)
            warning = "已存在保存的标注，本次只更新 Pred JSON"
        elif anno.exists() and not _review_json_is_empty(anno):
            out_path = _review_pred(cid)
            warning = "已有标注 JSON，本次只更新 Pred JSON；如需用模型结果覆盖，请先删除/备份标注"
        else:
            # 新整理出来的图片：模型结果直接写入 <id>.json，保持“一图一 JSON”
            out_path = anno
        pred = infer_image(img, cfg, out_path)
        npins = sum(len(v or {}) for v in (pred.get("pins") or {}).values())
        return jsonify({
            "ok": True,
            "case_id": cid,
            "components": len(pred.get("components") or {}),
            "pins": npins,
            "nets": len(pred.get("nets") or {}),
            "seconds": round(time.time() - t0, 1),
            "path": str(out_path),
            "warning": warning,
        })
    except Exception as exc:
        return jsonify({"error": str(exc), "trace": traceback.format_exc()[-4000:]}), 500
    finally:
        _PREDICT_LOCK.release()


@app.route("/api/review/promote/<cid>", methods=["POST"])
def review_promote(cid):
    mode = "review"
    d, img, js = _files(cid, mode)
    if img is None or d is None:
        return jsonify({"error": f"review case {cid} not found"}), 404
    if js is None:
        return jsonify({"error": "该案例还没有 Pred/标注 JSON"}), 400
    try:
        data = json.loads(js.read_text(encoding="utf-8"))
    except Exception as exc:
        return jsonify({"error": f"JSON 读取失败：{exc}"}), 400

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name) for p in TRAIN_DIR.iterdir() if p.is_dir() and p.name.isdigit()]
    train_id = None
    if cid.isdigit() and 1 <= int(cid) <= 9999:
        cand = f"{int(cid):04d}"
        if not (TRAIN_DIR / cand).exists():
            train_id = cand
    if train_id is None:
        nxt = max(existing) + 1 if existing else 1
        train_id = f"{nxt:04d}"
    dst = TRAIN_DIR / train_id
    if dst.exists():
        return jsonify({"error": f"训练集目录已存在：{dst}"}), 409
    dst.mkdir(parents=True)
    dst_img = dst / f"{train_id}{img.suffix.lower()}"
    shutil.copy2(img, dst_img)
    dst_json = dst / f"{train_id}_target.json"
    _write_json(dst_json, data)
    return jsonify({"ok": True, "train_id": train_id, "dir": str(dst),
                    "image": str(dst_img), "json": str(dst_json)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("APP_PORT", "5000")), debug=False)
