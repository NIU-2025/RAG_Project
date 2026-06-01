from app.parsers.base import BaseParser, ParsedDocument, ParsedPage
from app.services.ocr import OCRService
from app.core.logger import logger


class ImageParser(BaseParser):
    def parse(self, file_path: str) -> ParsedDocument:
        logger.info(f"开始 OCR 处理图片: {file_path}")
        text = OCRService.recognize(file_path)
        return ParsedDocument(
            pages=[ParsedPage(page_num=1, content=text)],
            metadata={"parser": "image_ocr"},
        )
