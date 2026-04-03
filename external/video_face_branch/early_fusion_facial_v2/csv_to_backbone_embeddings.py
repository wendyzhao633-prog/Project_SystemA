"""
从图片清单 CSV 批量提取 backbone 特征，输出 embeddings 与新 manifest。

典型场景：
  1) 先用 video_to_images_for_fusion 生成图片 + fusion_manifest.csv
  2) 再用本脚本把每张图转成 .npy 特征，供 early fusion 系统直接消费

示例（在 emonet 根目录）：
  python -m emonet.csv_to_backbone_embeddings \
      --input_csv "D:/Ravdess/fusion_manifest.csv" \
      --backbone_ckpt "D:/Computervision/emonet/ravdess_runs_tune_focus_r50/best_backbone.pth" \
      --output_manifest "D:/Ravdess/fusion_manifest_with_emb.csv" \
      --embeddings_dir "D:/Ravdess/fusion_embeddings" \
      --backbone resnet50 --batch_size 64 --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch

from emonet.ravdess_fusion_backbone import (
    build_backbone_only,
    extract_features,
    load_backbone_for_fusion,
    load_backbone_from_full_model_ckpt,
)


def _detect_path_col(fieldnames: Sequence[str]) -> str:
    for c in ("image_path", "path", "file_path"):
        if c in fieldnames:
            return c
    raise ValueError(f"未找到图片路径列（image_path/path/file_path），当前列: {list(fieldnames)}")


def _resolve_image_path(raw: str, image_dir: Path | None) -> Path:
    p = Path(str(raw).strip())
    if not p.is_absolute() and image_dir is not None:
        p = image_dir / p
    return p


def _read_rgb_256(path: Path, image_size: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"无法读取图像: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return rgb


def _sid_from_row(row: Dict[str, str], idx: int, image_path: Path) -> str:
    sid = str(row.get("sample_id", "")).strip()
    if sid:
        return sid
    return image_path.stem if image_path.stem else f"row_{idx:08d}"


def main() -> int:
    ap = argparse.ArgumentParser(description="CSV -> backbone embeddings + manifest")
    ap.add_argument("--input_csv", type=str, required=True, help="输入图片清单 CSV")
    ap.add_argument(
        "--image_dir",
        type=str,
        default="",
        help="当 CSV 路径列为相对路径时使用；为空则按当前工作目录解析",
    )
    ap.add_argument("--backbone_ckpt", type=str, required=True, help="best_backbone.pth 或 best_uar.pth")
    ap.add_argument(
        "--ckpt_type",
        type=str,
        default="backbone",
        choices=["backbone", "full_model"],
        help="backbone=仅backbone权重；full_model=完整FERsystem权重（自动抽 backbone.*）",
    )
    ap.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    ap.add_argument("--backbone_pretrained", action="store_true", help="需与训练结构一致")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--device", type=str, default=None, help="cuda:0 或 cpu；默认自动")
    ap.add_argument("--embeddings_dir", type=str, required=True, help="输出 .npy 特征目录")
    ap.add_argument("--output_manifest", type=str, required=True, help="输出清单 CSV（附 embedding_path）")
    ap.add_argument("--strict", action="store_true", help="遇到坏图/丢图立即退出；默认跳过并记录")
    args = ap.parse_args()

    in_csv = Path(args.input_csv)
    if not in_csv.is_file():
        print(f"找不到输入 CSV: {in_csv}", file=sys.stderr)
        return 2

    image_dir = None
    if str(args.image_dir).strip():
        image_dir = Path(args.image_dir)
        if not image_dir.is_dir():
            print(f"--image_dir 不是目录: {image_dir}", file=sys.stderr)
            return 2
        image_dir = image_dir.resolve()

    emb_dir = Path(args.embeddings_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(args.output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.ckpt_type == "backbone":
        bb = load_backbone_for_fusion(
            args.backbone_ckpt,
            backbone=args.backbone,
            backbone_pretrained=bool(args.backbone_pretrained),
            device=device,
        )
    else:
        bb = load_backbone_from_full_model_ckpt(
            args.backbone_ckpt,
            backbone=args.backbone,
            backbone_pretrained=bool(args.backbone_pretrained),
            device=device,
        )

    _, feat_dim = build_backbone_only(
        backbone=args.backbone,
        backbone_pretrained=bool(args.backbone_pretrained),
    )
    print(f"device={device} | backbone={args.backbone} | feat_dim={feat_dim}")

    kept_rows: List[Dict[str, str]] = []
    bad_rows: List[Tuple[int, str, str]] = []

    with in_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("输入 CSV 无表头", file=sys.stderr)
            return 2
        fieldnames_in = list(reader.fieldnames)
        path_col = _detect_path_col(fieldnames_in)
        rows = list(reader)

    # 批量推理
    batch_imgs: List[torch.Tensor] = []
    batch_meta: List[Tuple[int, Dict[str, str], Path, str]] = []
    n_saved = 0

    def flush_batch() -> None:
        nonlocal n_saved, kept_rows
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs, dim=0).to(device)
        with torch.no_grad():
            feats = extract_features(bb, x, flatten=True).cpu().numpy().astype(np.float32)
        for j, (idx, row, img_path, sid) in enumerate(batch_meta):
            rel_emb = Path(sid + ".npy")
            np.save(emb_dir / rel_emb, feats[j])
            out_row = dict(row)
            out_row["embedding_path"] = str((emb_dir / rel_emb).resolve())
            out_row["feature_dim"] = str(int(feats[j].shape[0]))
            out_row["resolved_image_path"] = str(img_path.resolve())
            kept_rows.append(out_row)
            n_saved += 1
        batch_imgs.clear()
        batch_meta.clear()

    for i, row in enumerate(rows, start=2):
        raw = row.get(path_col, "")
        img_path = _resolve_image_path(raw, image_dir)
        if not img_path.is_file():
            msg = "image_not_found"
            if args.strict:
                print(f"[ERROR] 行{i}: {msg} -> {img_path}", file=sys.stderr)
                return 1
            bad_rows.append((i, str(img_path), msg))
            continue
        try:
            rgb = _read_rgb_256(img_path, int(args.image_size))
        except Exception as e:  # noqa: BLE001
            if args.strict:
                print(f"[ERROR] 行{i}: read_failed -> {img_path} | {e}", file=sys.stderr)
                return 1
            bad_rows.append((i, str(img_path), f"read_failed:{e}"))
            continue

        x = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        sid = _sid_from_row(row, i, img_path)
        batch_imgs.append(x)
        batch_meta.append((i, row, img_path, sid))
        if len(batch_imgs) >= int(args.batch_size):
            flush_batch()

    flush_batch()

    # 输出 manifest
    fieldnames_out = fieldnames_in + ["resolved_image_path", "embedding_path", "feature_dim"]
    with out_manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_out)
        w.writeheader()
        w.writerows(kept_rows)

    print(f"输入行数: {len(rows)}")
    print(f"成功导出特征: {n_saved}")
    if bad_rows:
        print(f"跳过行数: {len(bad_rows)}")
        bad_path = out_manifest.with_suffix(".skipped.csv")
        with bad_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["line_no", "image_path", "reason"])
            w.writerows(bad_rows)
        print(f"跳过详情: {bad_path}")
    print(f"输出 manifest: {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
