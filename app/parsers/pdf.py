from typing import List
from app.parsers.base import BaseParser, ParsedDocument, ParsedPage, ContentBlock


class PDFParser(BaseParser):
    def parse(self, file_path: str) -> ParsedDocument:
        import fitz

        doc = ParsedDocument()
        pdf = fitz.open(file_path)
        prev_last_block = None  # 用于跨页表格合并

        for page_num, page in enumerate(pdf, start=1):
            blocks = []

            # 1. 纯文本
            text = self.clean_text(page.get_text("text"))
            if text:
                blocks.append(ContentBlock(type="text", content=text))

            # 2. 表格（结构化 pipe 渲染）
            table_contents = self._extract_tables(page)
            for tbl_text in table_contents:
                blocks.append(ContentBlock(type="table", content=tbl_text))

            # 3. 图片 OCR
            ocr_texts = self._extract_image_text(page, file_path)
            for ocr_text in ocr_texts:
                blocks.append(ContentBlock(type="image_text", content=ocr_text))

            # === 跨页表格合并 ===
            # 条件：上一页最后一块是 table，且当前页第一块也是 table，则合并为同一张表
            if prev_last_block and prev_last_block.type == "table" and blocks and blocks[0].type == "table":
                prev_last_block.content += "\n" + blocks.pop(0).content

            # 构建向后兼容的 content 字符串
            page_content = "\n\n".join(b.content for b in blocks)
            page_content = self.clean_text(page_content)

            if page_content or blocks:
                doc.pages.append(ParsedPage(
                    page_num=page_num,
                    content=page_content,
                    blocks=blocks,
                ))

            prev_last_block = blocks[-1] if blocks else None

        doc.metadata["total_pages"] = len(pdf)
        pdf.close()
        return doc

    def _extract_tables(self, page) -> List[str]:
        """提取页面中的表格，返回每个表格的 pipe 渲染文本列表"""
        try:
            tables = page.find_tables()
            if not tables or not tables.tables:
                return []
            results = []
            for tbl in tables.tables:
                rows = []
                for row in tbl.extract():
                    cells = [str(cell).strip() if cell is not None else "" for cell in row]
                    rows.append(" | ".join(cells))
                if rows:
                    results.append("\n".join(rows))
            return results
        except Exception:
            return []

    def _extract_image_text(self, page, file_path: str) -> List[str]:
        """提取页面中嵌入图片的 OCR 文字，返回文本列表"""
        try:
            images = page.get_images(full=True)
            if not images:
                return []
            from app.parsers.ocr_utils import ocr_image, clean_ocr_text
            import fitz

            results = []
            for img in images:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                if not base_image or not base_image.get("image"):
                    continue
                text = ocr_image(base_image["image"])
                text = clean_ocr_text(text)
                if text and len(text) > 2:
                    results.append(text)

            return results
        except Exception:
            return []
