"""Detect faces with YuNet and blur them in one image."""

import argparse
from pathlib import Path

import cv2 as cv

from yunet import YuNet


def blur_faces(image, detections):
    """Return a copy with each detected face blurred."""
    output = image.copy()
    image_height, image_width = output.shape[:2]

    for face in detections:
        x, y, width, height = (int(value) for value in face[:4])

        # Leave a small margin so the blur covers the edges of the face.
        padding_x = round(width * 0.1)
        padding_y = round(height * 0.1)
        left = max(0, x - padding_x)
        top = max(0, y - padding_y)
        right = min(image_width, x + width + padding_x)
        bottom = min(image_height, y + height + padding_y)

        region = output[top:bottom, left:right]
        if region.shape[0] < 3 or region.shape[1] < 3:
            continue

        # An odd, face-sized kernel keeps the blur visible at different scales.
        kernel = max(3, min(region.shape[:2]) // 2)
        if kernel % 2 == 0:
            kernel += 1
        output[top:bottom, left:right] = cv.GaussianBlur(region, (kernel, kernel), 0)

    return output


def main():
    parser = argparse.ArgumentParser(description="使用 YuNet 检测并模糊图片中的人脸")
    parser.add_argument("--input", required=True, help="输入图片路径")
    parser.add_argument("--output", default="blurred_result.jpg", help="输出图片路径")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent / "face_detection_yunet_2026may.onnx"),
        help="YuNet 模型路径",
    )
    parser.add_argument("--confidence", type=float, default=0.9, help="检测置信度，范围 0 到 1")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        parser.error("输入和输出不能是同一个文件")
    if not 0 < args.confidence <= 1:
        parser.error("--confidence 必须大于 0 且不超过 1")

    image = cv.imread(str(input_path))
    if image is None:
        parser.error(f"无法读取图片：{input_path}")

    height, width = image.shape[:2]
    model = YuNet(modelPath=args.model, inputSize=[width, height], confThreshold=args.confidence)
    detections = model.infer(image)
    result = blur_faces(image, detections)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(output_path), result):
        parser.error(f"无法保存图片：{output_path}")
    print(f"检测到 {len(detections)} 张人脸，结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
