import os
from app.parsers.base import BaseParser, ParsedDocument, ParsedPage


class PDFParser(BaseParser):
    def parse(self, file_path: str) -> ParsedDocument:
        import fitz

        doc = ParsedDocument()
        pdf = fitz.open(file_path)

        for page_num, page in enumerate(pdf, start=1):
            page_content = ""

            text = self.clean_text(page.get_text("text"))
            if text:
                page_content += text

            tables = self._extract_tables(page, page_num)
            if tables:
                page_content += "\n\n" + tables

            ocr_text = self._extract_image_text(page, page_num, file_path)
            if ocr_text:
                page_content += "\n\n" + ocr_text

            page_content = self.clean_text(page_content)
            if page_content:
                doc.pages.append(ParsedPage(page_num=page_num, content=page_content))

        doc.metadata["total_pages"] = len(pdf)
        pdf.close()
        return doc

    def _extract_tables(self, page, page_num: int) -> str:
        try:
            tables = page.find_tables()
            if not tables or not tables.tables:
                return ""
            parts = []
            for tbl in tables.tables:
                rows = []
                for row in tbl.extract():
                    cells = [str(cell).strip() if cell is not None else "" for cell in row]
                    rows.append(" | ".join(cells))
                if rows:
                    markdown = "\n".join(rows)
                    parts.append(f"【表格 第{page_num}页】\n{markdown}")
            return "\n\n".join(parts)
        except Exception:
            return ""

    def _extract_image_text(self, page, page_num: int, file_path: str) -> str:
        try:
            images = page.get_images(full=True)
            if not images:
                return ""
            from app.parsers.ocr_utils import ocr_image, clean_ocr_text
            import fitz

            parts = []
            for img in images:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                if not base_image or not base_image.get("image"):
                    continue
                text = ocr_image(base_image["image"])
                text = clean_ocr_text(text)
                if text and len(text) > 2:
                    parts.append(f"【图片文字 第{page_num}页】\n{text}")

            return "\n\n".join(parts)
        except Exception:
            return ""
