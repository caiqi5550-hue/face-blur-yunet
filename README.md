# 批量人脸检测与模糊处理

一个基于 OpenCV YuNet 的 Python 入门项目：检测图片中的人脸，对检测区域做高斯模糊，并批量处理文件夹中的图片。批量模式还会生成 CSV 统计表。

## 功能

- 单张图片：检测并模糊人脸，保存新图片。
- 批量图片：处理输入文件夹第一层的 JPG、JPEG、PNG、BMP、WebP 图片。
- 统计结果：在 `results.csv` 中记录输入文件名、检测到的人脸数、输出文件名和处理耗时（毫秒）。
- 可调节 `--confidence` 检测置信度；降低阈值可能减少漏检，也可能增加误检。

## 环境与安装

本项目在 Python 3.14、OpenCV 5.0.0.93、NumPy 2.5.4 下验证。

```bash
python3 -m pip install -r requirements.txt
```

## 运行

在本项目目录中执行。先将自己的测试图片放入 `input_images/`；图片及输出文件默认不纳入 Git 版本管理。

处理单张图片：

```bash
python3 face_blur.py --input input_images/test.jpg --output output_images/test_blurred.jpg
```

批量处理：

```bash
python3 batch_blur.py --input-dir input_images --output-dir output_images --confidence 0.5
```

处理完成后，在 `output_images/` 查看带 `_blurred` 后缀的图片及 `results.csv`。运行 `python3 face_blur.py --help` 或 `python3 batch_blur.py --help` 可查看全部参数。

## 实现方法

1. `yunet.py` 加载 YuNet ONNX 模型，返回人脸框与关键点。
2. `face_blur.py` 根据每个人脸框截取区域，向四周扩展约 10%，再应用 `cv2.GaussianBlur`。
3. `batch_blur.py` 逐张读取图片，复用模糊函数，将结果写入图片和 CSV。

本地练习中，同一张合照在置信度 0.9 下检测到 8 张人脸，在 0.5 下检测到 10 张。测试图片因使用权限不随仓库提供；这些数字只是本地观察结果，不是公开数据集上的精度指标。

## 局限

模型未检测到的人脸不会被模糊，因此输出图片需要人工检查。这个项目调用预训练模型进行推理，没有训练或改进 YuNet 模型。`elapsed_ms` 是每张图片的读取、推理、模糊及保存耗时，不包含模型初次加载时间。

## 来源与许可

本项目基于 [OpenCV Zoo 的 YuNet 人脸检测示例](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) 开发。`yunet.py`、ONNX 模型及 `LICENSE` 来自该项目；本项目新增了 `face_blur.py` 和 `batch_blur.py`，用于模糊、批量处理和统计。保留原项目的 MIT 许可与版权信息。
