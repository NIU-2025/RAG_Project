from app.parsers.base import BaseParser, ParsedDocument, ParsedPage


class MarkdownParser(BaseParser):
    """
    Markdown 解析器 — 保留原始 Markdown 结构。
    与 TextParser 的区别：不调用 clean_text()，避免 line.strip() 破坏
    缩进语义（嵌套列表、代码块等）。
    """

    def parse(self, file_path: str) -> ParsedDocument:
        doc = ParsedDocument()
        for encoding in ("utf-8", "gbk", "utf-16"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    text = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        # 只修复编码乱码，不破坏 Markdown 结构
        import ftfy
        text = ftfy.fix_text(text)
        text = text.strip()

        if text:
            doc.pages.append(ParsedPage(page_num=1, content=text))
        return doc
