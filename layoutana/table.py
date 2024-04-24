from layoutana.bbox import merge_boxes
from layoutana.utils import (
    convert_doc_to_pixel_coords,
    get_image_base64,
    get_image_bytes,
    save_debug_info,
    merge_target_blocks,
    set_block_type,
    set_special_block_type,
)
from layoutana.schema import Line, Span, Block, Page, TableInfo
from layoutana.settings import settings

from tabulate import tabulate
from typing import List
import io


def merge_table_caption(block_idx, block, page):
    prev_block: Block = None
    next_block: Block = None
    if block_idx > 0:
        prev_block = page.blocks[block_idx - 1]
    if block_idx < len(page.blocks) - 1:
        next_block = page.blocks[block_idx + 1]

    prev_block_type = prev_block.most_common_block_type() if prev_block else None
    next_block_type = next_block.most_common_block_type() if next_block else None

    if (prev_block_type is None and next_block_type is None) or (
        prev_block_type != "Caption" and next_block_type != "Caption"
    ):
        # very nesssary!!!
        return False, block.bbox, block.bbox

    table_caption_bbox: List[float] = block.bbox
    if prev_block_type is not None and "Table" in prev_block.prelim_text:
        # merge previous caption
        table_caption_bbox = merge_boxes(block.bbox, prev_block.bbox)
        set_block_type(prev_block, "TableCaption")
    elif next_block_type is not None and "Table" in next_block.prelim_text:
        # merge next caption
        table_caption_bbox = merge_boxes(block.bbox, next_block.bbox)
        set_block_type(next_block, "TableCaption")
    else:
        # has caption block, but not table
        set_block_type(block, "Picture")
        return False, None, None
    return True, table_caption_bbox, block.bbox


class CellRange:
    def __init__(self, cell_start: int, cell_end: int):
        self.cell_start = cell_start
        self.cell_end = cell_end

    def is_intersect(self, in_start: int, in_end: int, ratio: float = 0.0):
        real_ratio = self.intersect_ratio(in_start, in_end)
        return real_ratio > ratio

    def is_in_border(self, in_start: int, in_end: int, expect_gap: int = 2):
        if in_start - self.cell_end > 0:
            real_gap = in_start - self.cell_end
        elif self.cell_start - in_end > 0:
            real_gap = self.cell_start - in_end
        else:
            return False

        if real_gap < expect_gap:
            return True
        else:
            return False

    def intersect_ratio(self, in_start: int, in_end: int):
        in_length = in_end - in_start
        self_length = self.cell_end - self.cell_start
        intersect: bool = self.cell_start <= in_end and self.cell_end >= in_start
        if not intersect:
            return 0
        intersect_length = min(self.cell_end, in_end) - max(self.cell_start, in_start)
        return intersect_length / min(self_length, in_length)

    def extend(self, in_start: int, in_end: int):
        self.cell_start = min(self.cell_start, in_start)
        self.cell_end = max(self.cell_end, in_end)

    def is_in(self, in_start: int, in_end: int):
        return self.cell_start <= in_start and self.cell_end >= in_end


def calculate_column(column_x_list: List[CellRange], x_start: int, x_end: int):
    for column in column_x_list:
        if column.is_intersect(x_start, x_end, 0.0) or column.is_in_border(
            x_start, x_end, 5
        ):
            column.extend(x_start, x_end)
            return
    column_x_list.append(CellRange(x_start, x_end))


def calculate_row(row_y_list: List[CellRange], y_start: int, y_end: int):
    for row in row_y_list:
        if row.is_intersect(y_start, y_end, 0.7):
            row.extend(y_start, y_end)
            return
    row_y_list.append(CellRange(y_start, y_end))


def recognition_table(block: Block, table_idx: int, debug_mode):
    # get column
    column_x_list: List[CellRange] = []
    for line in block.lines:
        calculate_column(column_x_list, line.x_start, line.x_start + line.width)
    if column_x_list == []:
        return
    column_x_list.sort(key=lambda x: x.cell_start)

    # get row heads
    # row_y_list: List[CellRange] = []
    # column_one = column_x_list[0]
    # for line in block.lines:
    #     if column_one.is_in(line.x_start, line.x_start + line.width):
    #         row_y_list.append(CellRange(line.y_start, line.y_start + line.height))
    # if row_y_list == []:
    #     return
    # row_y_list.sort(key=lambda x: x.cell_start)

    # get row
    row_y_list: List[CellRange] = []
    for line in block.lines:
        calculate_row(row_y_list, line.y_start, line.y_start + line.height)
    if row_y_list == []:
        return
    row_y_list.sort(key=lambda x: x.cell_start)

    # merget row list
    # merge_row_list: List[CellRange] = []
    # for row in row_y_list:
    #     calculate_row(merge_row_list, row.cell_start, row.cell_end)

    table_matrix = []
    line_cache: dict = {}
    for row in row_y_list:
        table_row = []
        for column in column_x_list:
            current_cell: str = ""
            for line_index, line in enumerate(block.lines):
                if (
                    row.is_in(line.y_start, line.y_start + line.height)
                    and column.is_in(line.x_start, line.x_start + line.width)
                    and line_index not in line_cache
                ):
                    line_cache[line_index] = line
                    if current_cell == "":
                        current_cell = line.prelim_text
                    else:
                        current_cell = current_cell + " " + line.prelim_text

            table_row.append(current_cell)
        table_matrix.append(table_row)

    return table_to_markdown(table_matrix, block, table_idx)


def table_to_markdown(table: List[List[str]], block: Block, table_idx: int):
    # convert table matrix to markdown
    new_text = tabulate(table, tablefmt="github")
    block.lines = [
        Line(
            bbox=block.bbox,
            spans=[
                Span(
                    bbox=block.bbox,
                    span_id=f"{table_idx}_fix_table",
                    font="Table",
                    color=0,
                    block_type="Table",
                    text=f"{new_text}",
                )
            ],
        )
    ]
    return new_text


def detect_tables(pages: List[Page], debug_mode: bool = False):
    merge_target_blocks(pages, "Table")
    table_idx = 0

    tables_info: list[TableInfo] = []
    for page_idx, page in enumerate(pages):
        for block_idx, block in enumerate(page.blocks):
            if block.most_common_block_type() != "Table":
                # not table block
                set_special_block_type(block, "Table", "ErrorType")
                continue

            is_success, table_caption_bbox, table_bbox = merge_table_caption(
                block_idx, block, page
            )
            if not is_success:
                continue

            image_bytes = get_image_bytes(pages[page_idx], table_bbox)
            if image_bytes is None:
                continue

            # recognition_table_pdfplumber(pdfp.pages[page_idx], table_bbox, block, table_idx, True)
            table_text = recognition_table(block, table_idx, debug_mode)

            # table pixel bbox
            pixel_bbox = convert_doc_to_pixel_coords(table_bbox, settings.NOUGAT_DPI)
            block.table_pixel_bbox = pixel_bbox

            # save result
            if debug_mode:
                save_debug_info(
                    image_bytes,
                    "table",
                    page_idx,
                    block_idx,
                    [table_text, "predictions"],
                )

            tables_info.append(
                TableInfo(
                    content_base64=get_image_base64(image_bytes),
                    page_idx=page_idx,
                    block_idx=block_idx,
                    table_text=table_text,
                )
            )
            table_idx += 1
    return tables_info
