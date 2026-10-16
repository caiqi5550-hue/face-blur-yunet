"""Blur detected faces in every image directly inside an input folder."""

import argparse
import csv
import time
from pathlib import Path

import cv2 as cv

from face_blur import blur_faces
from yunet import YuNet


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="批量检测并模糊文件夹中的人脸")
    parser.add_argument("--input-dir", default="input_images", help="输入图片文件夹")
    parser.add_argument("--output-dir", default="output_images", help="输出图片文件夹")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent / "face_detection_yunet_2026may.onnx"),
        help="YuNet 模型路径",
    )
    parser.add_argument("--confidence", type=float, default=0.9, help="检测置信度，范围 0 到 1")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        parser.error(f"输入文件夹不存在：{input_dir}")
    if input_dir.resolve() == output_dir.resolve():
        parser.error("输入和输出不能是同一个文件夹")
    if not 0 < args.confidence <= 1:
        parser.error("--confidence 必须大于 0 且不超过 1")

    image_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        print(f"没有找到图片。请把 JPG 或 PNG 图片直接放进 {input_dir}。")
        return

    model = YuNet(modelPath=args.model, confThreshold=args.confidence)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    total_faces = 0
    records = []

    for input_path in image_paths:
        started = time.perf_counter()
        image = cv.imread(str(input_path))
        if image is None:
            print(f"跳过无法读取的图片：{input_path.name}")
            continue

        height, width = image.shape[:2]
        model.setInputSize([width, height])
        detections = model.infer(image)
        result = blur_faces(image, detections)
        output_path = output_dir / f"{input_path.stem}_blurred{input_path.suffix}"
        if not cv.imwrite(str(output_path), result):
            print(f"保存失败：{output_path}")
            continue

        count = len(detections)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        processed += 1
        total_faces += count
        records.append({
            "input_file": input_path.name,
            "face_count": count,
            "output_file": output_path.name,
            "elapsed_ms": elapsed_ms,
        })
        print(f"{input_path.name}: {count} 张人脸 → {output_path}")

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["input_file", "face_count", "output_file", "elapsed_ms"])
        writer.writeheader()
        writer.writerows(records)

    print(f"完成：处理 {processed} 张图片，检测到 {total_faces} 张人脸。")
    print(f"统计表已保存到 {csv_path}")


if __name__ == "__main__":
    main()
