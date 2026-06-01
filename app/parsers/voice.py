from app.parsers.base import BaseParser, ParsedDocument, ParsedPage


class VoiceParser(BaseParser):
    """
    语音文件解析器：将音频文件通过 ASR 识别为文本。
    实际识别逻辑在 DocumentService.process_document 中调用 VoiceASRService 完成，
    此 Parser 返回的 full_text 即为 ASR 识别结果。
    """

    def __init__(self, asr_text: str = ""):
        self.asr_text = asr_text

    def parse(self, file_path: str) -> ParsedDocument:
        """直接返回 ASR 识别文本（实际识别在 DocumentService 中调用）"""
        return ParsedDocument(
            pages=[ParsedPage(page_num=1, content=self.asr_text)],
            metadata={"file_path": file_path},
        )
