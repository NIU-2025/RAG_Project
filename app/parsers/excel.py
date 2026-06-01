from app.parsers.base import BaseParser, ParsedDocument, ParsedPage


'''
        Excel 解析器，用于将 Excel 文件解析为文本格式。
        支持解析 Excel 文件中的所有工作表，每个工作表的解析结果作为文档的一个页面。
        解析结果包含工作表名称、表头和数据行，每个数据行用 | 连接。
        空值用空字符串填充。

        原始 Excel：
        姓名 部门 迟到次数 扣款金额 
        张三 技术部 3      200 
        李四 销售部 5      500

        解析后的文本：

            【表格：Sheet1】
        姓名 | 部门 | 迟到次数 | 扣款金额
        ----------------------------------------
        张三 | 技术部 | 3 | 200
        李四 | 销售部 | 5 | 500

'''

class ExcelParser(BaseParser):
    def parse(self, file_path: str) -> ParsedDocument:
        import pandas as pd

        doc = ParsedDocument()
        xl = pd.ExcelFile(file_path)

        for sheet_num, sheet_name in enumerate(xl.sheet_names, start=1):
            # 解析每个工作表，将数据转换为字符串类型
            # 并删除所有全为空的行（how="all"），用空字符串填充缺失值（fillna("")）
            df = xl.parse(sheet_name, dtype=str) #将当前 Sheet 解析为 DataFrame，所有列强制转为字符串类型
            df = df.dropna(how="all").fillna("")
            if df.empty:
                continue

            lines = [f"【表格：{sheet_name}】"]
            # 表头 用 | 连接所有列名作为表头，如 姓名 | 部门 | 迟到次数 | 扣款金额
            lines.append(" | ".join(str(c) for c in df.columns))
            lines.append("-" * 40)
            # 数据行 df.iterrows() ：逐行遍历 DataFrame，返回行索引和行数据
            for _, row in df.iterrows():
                row_text = " | ".join(str(v) for v in row.values)
                lines.append(row_text)

            content = self.clean_text("\n".join(lines))
            if content:
                doc.pages.append(ParsedPage(page_num=sheet_num, content=content))

        return doc

'''
        CSV 解析器，用于将 CSV 文件解析为文本格式。
        支持解析 CSV 文件中的所有数据行，每个数据行用 | 连接。
        空值用空字符串填充。
'''
class CSVParser(BaseParser):
    def parse(self, file_path: str) -> ParsedDocument:
        import pandas as pd

        doc = ParsedDocument()
        try:
            df = pd.read_csv(file_path, dtype=str, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, dtype=str, encoding="gbk")

        df = df.dropna(how="all").fillna("")
        lines = [" | ".join(str(c) for c in df.columns), "-" * 40]
        for _, row in df.iterrows():
            lines.append(" | ".join(str(v) for v in row.values))

        content = self.clean_text("\n".join(lines))
        if content:
            doc.pages.append(ParsedPage(page_num=1, content=content))
        return doc
