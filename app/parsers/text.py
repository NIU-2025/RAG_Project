from app.parsers.base import BaseParser, ParsedDocument, ParsedPage

'''
        如果所有编码都失败（循环正常结束，没有 break ），则使用 utf-8 编码并忽略无法解码的
        字符（ errors="ignore" ），确保至少能读到部分内容。
'''
class TextParser(BaseParser):
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

        content = self.clean_text(text)
        if content:
            doc.pages.append(ParsedPage(page_num=1, content=content))
        return doc
