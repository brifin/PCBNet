"""Scan Dataset300 and build train/val/test splits."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class Case:
    case_id: str
    root: Path
    image: Path
    cvjson: Path
    spice: Path | None = None
    eds: Path | None = None

    @property
    def style(self) -> str:
        name = self.image.stem.lower()
        for token in ("jlc", "kicad", "altium", "datasheet"):
            if token in name:
                return token
        return "unknown"


def _first(paths: Iterable[Path]) -> Path | None:
    items = list(paths)
    return items[0] if items else None


def scan_dataset(dataset_root: str | Path) -> list[Case]:
    root = Path(dataset_root)
    cases: list[Case] = []
    if not root.exists():
        return cases
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        cvjson = _first(d.glob("*_CVJsonStd.json"))
        image = _first(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if cvjson is None or image is None:
            continue
        cases.append(
            Case(
                case_id=d.name,
                root=d,
                image=image,
                cvjson=cvjson,
                spice=_first(d.glob("*.net")),
                eds=_first(d.glob("*.eds")),
            )
        )
    return cases


def make_splits(cases: list[Case], seed: int = 42, ratio: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> dict[str, list[str]]:
    ids = [c.case_id for c in cases]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * ratio[0])
    n_val = int(n * ratio[1])
    train = sorted(ids[:n_train])
    val = sorted(ids[n_train : n_train + n_val])
    test = sorted(ids[n_train + n_val :])
    if not test and ids:
        test = [train.pop()] if train else []
    return {"train": train, "val": val, "test": test, "all": sorted(ids)}


def load_or_create_splits(data_dir: Path, cases: list[Case], seed: int = 42, ratio=(0.8, 0.1, 0.1)) -> dict[str, list[str]]:
    path = data_dir / "splits.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    splits = make_splits(cases, seed=seed, ratio=ratio)
    path.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    return splits


def index_cases(cases: list[Case]) -> dict[str, Case]:
    index = {c.case_id: c for c in cases}
    for case in cases:
        index.setdefault(case.case_id.lstrip("0") or "0", case)
        if case.case_id.isdigit():
            index.setdefault(str(int(case.case_id)), case)
    return index


def parse_case_ids(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]
