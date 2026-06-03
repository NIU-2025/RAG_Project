from app.parsers.base import BaseParser, ParsedDocument, ParsedPage, ContentBlock
from app.parsers.pdf import PDFParser
from app.parsers.word import WordParser
from app.parsers.excel import ExcelParser, CSVParser
from app.parsers.text import TextParser
from app.parsers.image import ImageParser
from app.parsers.voice import VoiceParser


def get_parser(file_type: str, asr_text: str = "") -> BaseParser:
    mapping = {
        "pdf": PDFParser,
        "docx": WordParser,
        "doc": WordParser,
        "xlsx": ExcelParser,
        "xls": ExcelParser,
        "csv": CSVParser,
        "txt": TextParser,
        "md": TextParser,
        "png": ImageParser,
        "jpg": ImageParser,
        "jpeg": ImageParser,
        "gif": ImageParser,
        "bmp": ImageParser,
        "webp": ImageParser,
        "mp3": VoiceParser,
        "wav": VoiceParser,
        "m4a": VoiceParser,
        "aac": VoiceParser,
        "ogg": VoiceParser,
        "wma": VoiceParser,
        "flac": VoiceParser,
        "pcm": VoiceParser,
    }
    cls = mapping.get(file_type.lower())
    if not cls:
        raise ValueError(f"不支持的文件类型: {file_type}")
    if file_type.lower() in mapping and mapping[file_type.lower()] is VoiceParser:
        return cls(asr_text=asr_text)
    return cls()
