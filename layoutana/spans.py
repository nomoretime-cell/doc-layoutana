from layoutana.schema import Span

from enum import Enum
from typing import List, Dict, Optional


class SpanType(Enum):
    Caption = "Caption"
    Footnote = "Footnote"
    Formula = "Formula"
    ListItem = "List-item"
    PageFooter = "Page-footer"
    PageHeader = "Page-header"
    Picture = "Picture"
    SectionHeader = "Section-header"
    Table = "Table"
    Text = "Text"
    Title = "Title"


class FontSize:
    def __init__(self, font_size: int, spans_size: Optional[int] = 0):
        self.font_size = font_size
        self.spans_size = spans_size

    def update_spans_size(self, spans_size: int):
        self.spans_size = spans_size

    def inc_spans_size(self):
        self.spans_size += 1


class Font:
    def __init__(self, font: str, spans_size: Optional[int] = 0):
        self.font = font
        self.spans_size = spans_size

    def update_spans_size(self, spans_size: int):
        self.spans_size = spans_size

    def inc_spans_size(self):
        self.spans_size += 1


class SpansAnalyzer:
    def __init__(self, spans: List[Span]):
        self.typeSize2spans: Dict[str, List[Span]] = {}
        self.type2fontSize: Dict[str, List[FontSize]] = {}
        self.type2font: Dict[str, List[Font]] = {}

        for span in spans:
            # update typeSize2spans
            type_size_key = span.block_type + "_" + str(span.size)
            if type_size_key not in self.typeSize2spans:
                self.typeSize2spans[type_size_key] = []
            self.typeSize2spans[type_size_key].append(span)

            # update type2font
            if span.block_type not in self.type2font:
                self.type2font[span.block_type] = []
            contain_font = False
            for font in self.type2font[span.block_type]:
                if font.font == span.font:
                    contain_font = True
                    font.inc_spans_size()
            if not contain_font:
                self.type2font[span.block_type].append(Font(span.font, 1))

        # update type2fontSize
        for type_size, spans in self.typeSize2spans.items():
            type = type_size.split("_")[0]
            font_size = int(type_size.split("_")[1])
            if type not in self.type2fontSize:
                self.type2fontSize[type] = []
            self.type2fontSize[type].append(FontSize(font_size, len(spans)))

        # 对List[FontSize]按照spans_size字段降序排序
        for font_sizes in self.type2fontSize.values():
            font_sizes.sort(key=lambda x: x.spans_size, reverse=True)
        # 对List[Font]按照spans_size字段降序排序
        for font in self.type2font.values():
            font.sort(key=lambda x: x.spans_size, reverse=True)

        pass

    def get_type_to_fontsize(self) -> Dict[str, List[FontSize]]:
        return self.type2fontSize

    def get_type_to_font(self) -> Dict[str, List[Font]]:
        return self.type2font

    def get_typesize_to_spans(self) -> Dict[str, List[Span]]:
        return self.typeSize2spans

    def get_most_font_size(self, type: SpanType) -> int:
        return self.type2fontSize[type.value][0].font_size

    def get_most_font_type(self, type: SpanType) -> str:
        return self.type2font[type.value][0].font
