"""
从视频批量导出图片并生成清单 CSV，供 early fusion 系统使用。

能力：
1) 扫描目录中的视频（可递归）
2) 每个视频均匀抽取 N 帧
3) 可选做人脸裁剪（OpenCV Haar；失败时回退整帧缩放）
4) 生成图片 + manifest CSV
5) 若文件名符合 RAVDESS 规则，自动补充 label_id / label_name / actor

示例（在 emonet 根目录）：
  python -m emonet.video_to_images_for_fusion --video_dir "D:/Ravdess/videos" \
      --out_images_dir "D:/Ravdess/fusion_frames" --output_csv "fusion_manifest.csv" \
      --frames_per_video 8 --recursive --face_crop
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from emonet.ravdess_sample_id import normalize_ravdess_face_crop_stem, parse_ravdess_stem

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _iter_videos(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        vids = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    else:
        vids = [p for p in root.glob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(vids)


def _uniform_indices(n_frames: int, k: int) -> List[int]:
    if n_frames <= 0:
        return []
    if k <= 1:
        return [max(0, min(n_frames - 1, n_frames // 2))]
    kk = min(k, n_frames)
    pts = np.linspace(0, n_frames - 1, num=kk, dtype=float)
    idxs = sorted({max(0, min(n_frames - 1, int(round(x)))) for x in pts})
    return idxs


def _resize_square(img_bgr: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)


def _center_crop_square(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    side = min(h, w)
    y0 = max(0, (h - side) // 2)
    x0 = max(0, (w - side) // 2)
    return img_bgr[y0 : y0 + side, x0 : x0 + side]


def _largest_face_crop(
    frame_bgr: np.ndarray,
    face_detector: cv2.CascadeClassifier,
) -> Tuple[np.ndarray, bool]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    if len(faces) == 0:
        return frame_bgr, False
    x, y, w, h = max(faces, key=lambda b: int(b[2]) * int(b[3]))
    x0, y0 = max(0, x), max(0, y)
    x1 = min(frame_bgr.shape[1], x + w)
    y1 = min(frame_bgr.shape[0], y + h)
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return frame_bgr, False
    return crop, True


def _ravdess_meta_from_video(video_path: Path) -> Dict[str, str]:
    stem = normalize_ravdess_face_crop_stem(video_path.stem)
    meta = parse_ravdess_stem(stem)
    return {
        "sample_id": str(meta["sample_id"]),
        "label_id": str(meta["label_id"]),
        "label_name": str(meta["label_name"]),
        "actor": str(meta["actor"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="视频->图片+CSV（early fusion 输入）")
    ap.add_argument("--video_dir", type=str, required=True, help="视频目录")
    ap.add_argument("--out_images_dir", type=str, required=True, help="导出图片目录")
    ap.add_argument("--output_csv", type=str, default="fusion_manifest.csv", help="输出清单 CSV")
    ap.add_argument("--frames_per_video", type=int, default=8, help="每个视频抽帧数")
    ap.add_argument("--image_size", type=int, default=256, help="输出图片边长")
    ap.add_argument("--recursive", action="store_true", help="递归子目录")
    ap.add_argument("--face_crop", action="store_true", help="启用人脸裁剪（OpenCV Haar）")
    ap.add_argument(
        "--strict_ravdess",
        action="store_true",
        help="要求视频名必须可按 RAVDESS 七段解析；否则报错退出",
    )
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    if not video_dir.is_dir():
        print(f"--video_dir 不是目录: {video_dir}", file=sys.stderr)
        return 2

    out_img = Path(args.out_images_dir)
    out_img.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.output_csv)
    if not out_csv.is_absolute():
        out_csv = Path.cwd() / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    videos = _iter_videos(video_dir, args.recursive)
    if not videos:
        print(f"未找到视频: {video_dir}", file=sys.stderr)
        return 1

    detector = None
    if args.face_crop:
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if detector.empty():
            print("OpenCV Haar 模型加载失败。请检查 OpenCV 安装。", file=sys.stderr)
            return 1

    rows: List[Dict[str, str]] = []
    n_saved = 0
    n_face_ok = 0
    skipped: List[Tuple[str, str]] = []

    for vp in videos:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            skipped.append((str(vp), "open_failed"))
            continue
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idxs = _uniform_indices(n_frames, int(args.frames_per_video))
        if not idxs:
            cap.release()
            skipped.append((str(vp), "no_frames"))
            continue

        ravdess_meta = {"sample_id": vp.stem, "label_id": "", "label_name": "", "actor": ""}
        try:
            ravdess_meta = _ravdess_meta_from_video(vp)
        except ValueError as e:
            if args.strict_ravdess:
                cap.release()
                print(f"[ERROR] {vp.name}: {e}", file=sys.stderr)
                return 1

        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue

            face_ok = False
            crop = frame_bgr
            if detector is not None:
                crop, face_ok = _largest_face_crop(frame_bgr, detector)
                if not face_ok:
                    # 回退：整帧中心裁剪，保证流程不中断
                    crop = _center_crop_square(frame_bgr)
                else:
                    n_face_ok += 1
            else:
                crop = _center_crop_square(frame_bgr)

            img = _resize_square(crop, int(args.image_size))
            out_name = f"{vp.stem}_f{int(fi):06d}.png"
            out_path = out_img / out_name
            cv2.imwrite(str(out_path), img)

            rows.append(
                {
                    "image_path": str(out_path.resolve()),
                    "video_path": str(vp.resolve()),
                    "video_id": vp.stem,
                    "frame_idx": str(int(fi)),
                    "sample_id": f"{ravdess_meta['sample_id']}_f{int(fi):06d}",
                    "label_id": ravdess_meta["label_id"],
                    "label_name": ravdess_meta["label_name"],
                    "actor": ravdess_meta["actor"],
                    "ok_face": "1" if face_ok else "0",
                }
            )
            n_saved += 1

        cap.release()

    fieldnames = [
        "image_path",
        "video_path",
        "video_id",
        "frame_idx",
        "sample_id",
        "label_id",
        "label_name",
        "actor",
        "ok_face",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"视频总数: {len(videos)}")
    print(f"导出图片数: {n_saved}")
    if args.face_crop:
        print(f"成功人脸裁剪数: {n_face_ok} | 回退中心裁剪数: {max(0, n_saved - n_face_ok)}")
    if skipped:
        print(f"跳过视频: {len(skipped)}（打开失败或无帧）")
    print(f"已写入 CSV: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
