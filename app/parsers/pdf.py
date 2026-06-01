from app.parsers.base import BaseParser, ParsedDocument, ParsedPage


class PDFParser(BaseParser):
    def parse(self, file_path: str) -> ParsedDocument:
        import fitz  # PyMuPDF

        doc = ParsedDocument()
        pdf = fitz.open(file_path)
        for page_num, page in enumerate(pdf, start=1):
            #- "text" 参数表示提取纯文本格式
            #- - PyMuPDF 还支持 "html" 、 "dict" 、 "blocks" 等格式提取 
            text = page.get_text("text")
            cleaned = self.clean_text(text)
            if cleaned:
                doc.pages.append(ParsedPage(page_num=page_num, content=cleaned))
        doc.metadata["total_pages"] = len(pdf)
        pdf.close()
        return doc
