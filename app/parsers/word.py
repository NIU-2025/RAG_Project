from typing import List
from app.parsers.base import BaseParser, ParsedDocument, ParsedPage, ContentBlock


class WordParser(BaseParser):
    def parse(self, file_path: str) -> ParsedDocument:
        from docx import Document as DocxDocument
        from docx.oxml.ns import qn
        from docx.text.paragraph import Paragraph

        docx = DocxDocument(file_path)
        blocks: List[ContentBlock] = []

        # 按文档顺序遍历 body 元素（段落和表格交错）
        table_index = 0
        current_paragraphs: List[str] = []

        for child in docx.element.body:
            # 段落
            if child.tag == qn("w:p"):
                para = Paragraph(child, docx)
                text = para.text.strip()
                if text:
                    current_paragraphs.append(text)

            # 表格
            elif child.tag == qn("w:tbl"):
                # 先落盘此表之前的段落
                if current_paragraphs:
                    blocks.append(ContentBlock(
                        type="text",
                        content="\n".join(current_paragraphs),
                    ))
                    current_paragraphs = []

                # 提取表格，整表作为一个原子块
                if table_index < len(docx.tables):
                    table = docx.tables[table_index]
                    rows = []
                    for row in table.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        rows.append(" | ".join(cells))
                    if rows:
                        blocks.append(ContentBlock(
                            type="table",
                            content="\n".join(rows),
                        ))
                    table_index += 1

        # 落盘末尾的段落
        if current_paragraphs:
            blocks.append(ContentBlock(
                type="text",
                content="\n".join(current_paragraphs),
            ))

        # 构建向后兼容的 full_text
        full_text = "\n\n".join(b.content for b in blocks)
        full_text = self.clean_text(full_text)

        if full_text or blocks:
            doc = ParsedDocument()
            doc.pages.append(ParsedPage(
                page_num=1,
                content=full_text,
                blocks=blocks,
            ))
            return doc

        return ParsedDocument()
