from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

'''
这个文件定义了 文档解析器的基类和数据结构 ，是整个 RAG 系统中文档处理的第一步——
将各种格式的文件（PDF、Word、Excel 等）解析为统一的文本格式。
'''

@dataclass
class ParsedPage:
    page_num: int
    content: str

#表示 整个文档 的解析结果，包含所有页面和元数据
@dataclass
class ParsedDocument:
    pages: List[ParsedPage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # 合并所有页面内容为一个字符串 @property将方法转换为属性访问，方便后续使用
    @property
    def full_text(self) -> str:
        #将所有页面的内容用两个换行符连接
        return "\n\n".join(p.content for p in self.pages if p.content.strip())


class BaseParser(ABC):
    # 定义解析方法，必须实现
    @abstractmethod 
    def parse(self, file_path: str) -> ParsedDocument:
        pass

    # 定义文本清理方法，用于处理解析后的文本，去除空行和额外空白
    @staticmethod
    def clean_text(text: str) -> str:
        import ftfy
        import re

        #修复文本编码问题（如乱码字符）
        text = ftfy.fix_text(text)
        # 去除连续空行，最多保留两个空行间隔
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 去除行首尾空白
        lines = [line.strip() for line in text.splitlines()]
        # 过滤纯空行聚集
        text = "\n".join(lines)
        return text.strip()
