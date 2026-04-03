"""
将帧级 embedding 清单聚合为视频级序列特征。

输入：manifest_with_emb.csv（至少包含 video_id, frame_idx, embedding_path）
输出：
  1) 每个 video_id 一个 .npy，形状 (T, D)
  2) sequence_manifest.csv（记录每个视频序列文件、长度、特征维度、可选标签）

示例（在 emonet 根目录）：
  python -m emonet.manifest_emb_to_sequences \
      --input_manifest "D:/Ravdess/fusion_manifest_with_emb.csv" \
      --out_sequences_dir "D:/Ravdess/fusion_sequences" \
      --output_manifest "D:/Ravdess/sequence_manifest.csv"
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def _to_int_safe(v: str) -> int:
    s = str(v).strip()
    if s == "":
        return -1
    try:
        return int(s)
    except ValueError:
        m = re.search(r"\d+", s)
        return int(m.group(0)) if m else -1


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "video"


def _infer_video_id_and_frame_idx(row: dict) -> tuple[str, int]:
    """
    当 CSV 缺少 video_id/frame_idx 时，从 sample_id 或路径文件名推断。
    支持:
      - 03-01-...-12_f000123
      - 03-01-...-12
    """
    sid = str(row.get("sample_id", "")).strip()
    if not sid:
        for c in ("image_path", "path", "file_path", "resolved_image_path"):
            v = str(row.get(c, "")).strip()
            if v:
                sid = Path(v).stem
                break
    if not sid:
        return "unknown_video", -1

    m = re.match(r"^(.+)_f(\d+)$", sid)
    if m:
        return m.group(1), int(m.group(2))
    return sid, -1


def main() -> int:
    ap = argparse.ArgumentParser(description="manifest_with_emb.csv -> 视频级序列特征")
    ap.add_argument("--input_manifest", type=str, required=True, help="帧级 embedding 清单 CSV")
    ap.add_argument("--out_sequences_dir", type=str, required=True, help="输出视频序列 .npy 目录")
    ap.add_argument("--output_manifest", type=str, required=True, help="输出视频级清单 CSV")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="任一 embedding 缺失/损坏即退出；默认跳过坏行",
    )
    args = ap.parse_args()

    in_csv = Path(args.input_manifest)
    if not in_csv.is_file():
        print(f"找不到输入 manifest: {in_csv}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_sequences_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(args.output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    with in_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("输入 CSV 无表头", file=sys.stderr)
            return 2
        required = {"embedding_path"}
        miss = [k for k in required if k not in reader.fieldnames]
        if miss:
            print(f"输入 CSV 缺少列: {miss}", file=sys.stderr)
            return 2
        rows = list(reader)

    has_video_id = "video_id" in (rows[0].keys() if rows else [])
    has_frame_idx = "frame_idx" in (rows[0].keys() if rows else [])
    if not (has_video_id and has_frame_idx):
        print("[INFO] 输入缺少 video_id/frame_idx，已启用从 sample_id/路径自动推断。")

    groups: Dict[str, List[dict]] = {}
    for row in rows:
        if has_video_id and has_frame_idx:
            vid = str(row.get("video_id", "")).strip() or "unknown_video"
            frame_idx = _to_int_safe(str(row.get("frame_idx", "")))
        else:
            vid, frame_idx = _infer_video_id_and_frame_idx(row)
        row["_video_id_resolved"] = vid
        row["_frame_idx_resolved"] = str(frame_idx)
        groups.setdefault(vid, []).append(row)

    out_rows: List[dict] = []
    skipped_rows: List[dict] = []
    used_names: Dict[str, int] = {}

    for vid, items in groups.items():
        items.sort(key=lambda r: _to_int_safe(str(r.get("_frame_idx_resolved", ""))))
        seq_feats: List[np.ndarray] = []
        label_id = ""
        label_name = ""
        actor = ""

        for r in items:
            emb_p = Path(str(r.get("embedding_path", "")).strip())
            if not emb_p.is_absolute():
                emb_p = (in_csv.parent / emb_p).resolve()
            if not emb_p.is_file():
                if args.strict:
                    print(f"[ERROR] embedding 不存在: {emb_p}", file=sys.stderr)
                    return 1
                skipped_rows.append({"video_id": vid, "reason": "embedding_not_found", "path": str(emb_p)})
                continue
            try:
                feat = np.load(emb_p)
            except Exception as e:  # noqa: BLE001
                if args.strict:
                    print(f"[ERROR] embedding 读取失败: {emb_p} | {e}", file=sys.stderr)
                    return 1
                skipped_rows.append({"video_id": vid, "reason": f"embedding_read_failed:{e}", "path": str(emb_p)})
                continue

            feat = np.asarray(feat, dtype=np.float32)
            if feat.ndim == 2 and feat.shape[0] == 1:
                feat = feat[0]
            if feat.ndim != 1:
                if args.strict:
                    print(f"[ERROR] embedding 维度非法，期望 1D 或 (1,D): {emb_p} -> {feat.shape}", file=sys.stderr)
                    return 1
                skipped_rows.append({"video_id": vid, "reason": f"bad_shape:{feat.shape}", "path": str(emb_p)})
                continue

            seq_feats.append(feat)
            if not label_id:
                label_id = str(r.get("label_id", "")).strip()
            if not label_name:
                label_name = str(r.get("label_name", "")).strip()
            if not actor:
                actor = str(r.get("actor", "")).strip()

        if not seq_feats:
            continue

        # 对齐维度（理论上应一致）
        dim = int(seq_feats[0].shape[0])
        seq_feats = [x for x in seq_feats if int(x.shape[0]) == dim]
        if not seq_feats:
            continue

        seq = np.stack(seq_feats, axis=0).astype(np.float32)  # (T, D)
        safe = _sanitize_filename(vid)
        idx = used_names.get(safe, 0)
        used_names[safe] = idx + 1
        out_name = f"{safe}.npy" if idx == 0 else f"{safe}__{idx}.npy"
        out_path = out_dir / out_name
        np.save(out_path, seq)

        out_rows.append(
            {
                "video_id": vid,
                "sequence_path": str(out_path.resolve()),
                "num_frames": str(int(seq.shape[0])),
                "feature_dim": str(int(seq.shape[1])),
                "label_id": label_id,
                "label_name": label_name,
                "actor": actor,
            }
        )

    fieldnames = [
        "video_id",
        "sequence_path",
        "num_frames",
        "feature_dim",
        "label_id",
        "label_name",
        "actor",
    ]
    with out_manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"输入帧行数: {len(rows)}")
    print(f"输出视频序列数: {len(out_rows)}")
    print(f"序列目录: {out_dir.resolve()}")
    print(f"输出 manifest: {out_manifest.resolve()}")

    if skipped_rows:
        bad_path = out_manifest.with_suffix(".skipped.csv")
        with bad_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["video_id", "reason", "path"])
            w.writeheader()
            w.writerows(skipped_rows)
        print(f"跳过帧数: {len(skipped_rows)} | 详情: {bad_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
