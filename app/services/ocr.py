import re
import os
from PIL import Image
from rapidocr import RapidOCR
from app.core.logger import logger

# RapidOCR 模型在首次推理时自动下载到 ~/.rapidocr/，无需手动配置
# 若需要自定义模型路径，可通过环境变量 RAPIDOCR_MODEL_DIR 设置


class OCRService:
    _instance = None
    _engine = None

    # 获取 RapidOCR 识别器实例，确保单例模式
    @classmethod
    def get_engine(cls) -> RapidOCR:
        if cls._engine is None:
            logger.info("初始化 RapidOCR 识别器（中英文）")
            cls._engine = RapidOCR()
            logger.info("RapidOCR 识别器初始化完成")
        return cls._engine

    @staticmethod
    def _is_valid_image(image_path: str) -> bool:
        """校验图片文件是否有效且非空"""
        try:
            if not os.path.exists(image_path):
                return False
            if os.path.getsize(image_path) == 0:
                logger.warning(f"图片文件为空: {image_path}")
                return False
            img = Image.open(image_path)
            img.verify()
            img = Image.open(image_path)
            width, height = img.size
            if width < 10 or height < 10:
                logger.warning(f"图片尺寸过小 ({width}x{height}): {image_path}")
                return False
            return True
        except Exception as e:
            logger.warning(f"图片文件损坏或无法读取 {image_path}: {e}")
            return False

    @staticmethod
    def clean_text(text: str) -> str:
        """对 OCR 识别结果进行清洗：去除控制字符、规范化空白、合并换行"""
        if not text:
            return ""

        # 1. 去除不可见控制字符
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)

        # 2. 将所有连续空白字符（含 tab 等）替换为单个空格
        text = re.sub(r"[ \t]+", " ", text)

        # 3. 按行分割，逐行清洗
        raw_lines = text.splitlines()
        cleaned = []
        for raw in raw_lines:
            line = raw.strip()
            if not line:
                cleaned.append("")
            else:
                line = re.sub(r" +", " ", line)
                cleaned.append(line)

        # 4. 去除首尾空行
        while cleaned and cleaned[0] == "":
            cleaned.pop(0)
        while cleaned and cleaned[-1] == "":
            cleaned.pop()

        # 5. 合并连续空行（最多保留一个空行作为段落分隔）
        merged = []
        for line in cleaned:
            if line == "":
                if merged and merged[-1] != "":
                    merged.append("")
            else:
                merged.append(line)

        # 6. 去除末尾空行
        while merged and merged[-1] == "":
            merged.pop()

        return "\n".join(merged).strip()

    @classmethod
    def recognize(cls, image_path: str) -> str:
        """
        使用 RapidOCR 识别图片中的文字。

        异常情况处理：
        - 文件不存在 / 文件为空 / 图片损坏  → raise ValueError("图片文件无效")
        - 图片尺寸过小                     → raise ValueError("图片文件无效")
        - 未识别到任何文字                  → raise ValueError("图片中未识别到文字内容")
        - 其他 OCR 异常                    → raise ValueError(f"OCR 处理失败: {原错误}")
        """
        # 前置校验：文件有效性
        if not os.path.exists(image_path):
            raise ValueError(f"图片文件不存在: {image_path}")
        if os.path.getsize(image_path) == 0:
            raise ValueError("图片文件为空")
        try:
            img = Image.open(image_path)
            img.verify()
            img = Image.open(image_path)
            if img.size[0] < 10 or img.size[1] < 10:
                raise ValueError("图片文件无效")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"图片文件无效: {e}")

        try:
            engine = cls.get_engine()
            output = engine(image_path)

            if output is None or not hasattr(output, "txts"):
                logger.warning(f"RapidOCR 未识别到任何内容: {image_path}")
                raise ValueError("图片中未识别到文字内容")

            txts = output.txts
            if not txts:
                logger.warning(f"RapidOCR 未识别到任何内容: {image_path}")
                raise ValueError("图片中未识别到文字内容")

            raw_text = " ".join(str(t) for t in txts)
            cleaned = cls.clean_text(raw_text)

            if not cleaned:
                logger.warning(f"RapidOCR 清洗后文本为空: {image_path}")
                raise ValueError("图片中未识别到文字内容")

            elapsed = getattr(output, "elapse", 0)
            logger.info(f"RapidOCR 识别完成，提取文字 {len(cleaned)} 字符，耗时 {elapsed:.3f}s")
            return cleaned

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"RapidOCR 处理异常: {e}")
            raise ValueError(f"OCR 处理失败: {e}")
