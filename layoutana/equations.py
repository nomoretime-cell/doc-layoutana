from layoutana.spans import SpanType, SpansAnalyzer
from layoutana.bbox import is_in_same_line, merge_boxes
from layoutana.utils import get_image_base64, get_image_bytes, save_debug_info
from layoutana.schema import EquationInfo, Page, Line, Block
from typing import List, Tuple
import os
import io

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_page_equation_regions(page: Page):
    i = 0
    equation_blocks_index_set = set()
    equation_blocks_index_list: List[List[int]] = []

    # get all equation lines
    equation_lines_bbox: List[List[float]] = [
        b.bbox for b in page.blocks if b.most_common_block_type() == "Formula"
    ]
    if len(equation_lines_bbox) == 0:
        # current page do not contain equation
        return [], []

    # current page contain equation
    while i < len(page.blocks):
        # current block object
        block_obj = page.blocks[i]
        # check if the block contains an equation
        if not block_obj.contains_equation(equation_lines_bbox):
            i += 1
            continue

        # cache first equation
        equation_blocks: List[Tuple[int, Block]] = [(i, block_obj)]
        equation_block_text = block_obj.prelim_text

        # Merge surrounding blocks
        if i > 0:
            # Merge previous blocks
            j = i - 1
            prev_block = page.blocks[j]
            prev_bbox = prev_block.bbox
            while (
                (
                    is_in_same_line(prev_bbox, block_obj.bbox)
                    or prev_block.contains_equation(equation_lines_bbox)
                )
                and j >= 0
                and j not in equation_blocks_index_set
            ):
                # block_bbox = merge_boxes(prev_bbox, block_bbox)

                # check if tokens is overwhelm
                prelim_block_text = prev_block.prelim_text + " " + equation_block_text

                equation_block_text = prelim_block_text
                equation_blocks.append((j, prev_block))
                j -= 1
                if j >= 0:
                    prev_block = page.blocks[j]
                    prev_bbox = prev_block.bbox

        if i < len(page.blocks) - 1:
            # Merge subsequent blocks
            i = i + 1
            next_block = page.blocks[i]
            next_bbox = next_block.bbox
            while (
                is_in_same_line(block_obj.bbox, next_bbox)
                or next_block.contains_equation(equation_lines_bbox)
                or len(equation_blocks) <= 3
            ) and i <= len(page.blocks) - 1:
                # block_bbox = merge_boxes(block_bbox, next_bbox)

                # check if tokens is overwhelm
                prelim_block_text = equation_block_text + " " + next_block.prelim_text

                equation_block_text = prelim_block_text
                equation_blocks.append((i, next_block))
                i += 1
                if i <= len(page.blocks) - 1:
                    next_block = page.blocks[i]
                    next_bbox = next_block.bbox

        equation_blocks_index = sorted(([sb[0] for sb in equation_blocks]))
        # Get indices of all blocks to merge
        equation_blocks_index_list.append(equation_blocks_index)
        equation_blocks_index_set.update(equation_blocks_index)

        i += 1

    return equation_blocks_index_list


def get_bbox(page: Page, region: List[int]):
    block_bboxes: List[List[float]] = []
    merged_block_bbox: List[float] = None
    for idx in region:
        block = page.blocks[idx]
        bbox = block.bbox
        if merged_block_bbox is None:
            merged_block_bbox = bbox
        else:
            merged_block_bbox = merge_boxes(merged_block_bbox, bbox)
        block_bboxes.append(bbox)
    return block_bboxes, merged_block_bbox


def detect_equations(
    pages: List[Page],
    debug_mode: bool = False,
):
    equations_info: list[EquationInfo] = []

    # 1. Find potential equation regions, and length of text in each region
    doc_equation_region_list: List[List[List[int]]] = []
    for pnum, page in enumerate(pages):
        regions: List[List[int]] = []
        regions = get_page_equation_regions(page)
        doc_equation_region_list.append(regions)

    # 2. Get images for each region
    doc_equation_images: List[io.BytesIO] = []
    doc_merged_equation_bbox: List[List[float]] = []
    for page_idx, page_equation_region_index in enumerate(doc_equation_region_list):
        page = pages[page_idx]
        for index, equation_region_index in enumerate(page_equation_region_index):
            if equation_region_index == []:
                continue
            # foreach equation region in one page
            #   "equation_region_index" is a list of block indices
            equation_bboxes, merged_equation_bbox = get_bbox(
                pages[page_idx], equation_region_index
            )
            equation_image: io.BytesIO = get_image_bytes(page, merged_equation_bbox)

            doc_equation_images.append(equation_image)
            doc_merged_equation_bbox.append(merged_equation_bbox)

            equations_info.append(
                EquationInfo(
                    content_base64=get_image_base64(equation_image),
                    page_idx=page_idx,
                    block_idx=-1,
                )
            )

    # save result
    if debug_mode:
        for idx, image in enumerate(doc_equation_images):
            save_debug_info(image, "equations", page_idx, idx)

    for page_idx, page in enumerate(pages):
        if page_idx == 0:
            continue
        inline_equations_info = detect_inline_equations_v2(page_idx, page, debug_mode)
        equations_info.extend(inline_equations_info)

    return equations_info


def if_contain_equation_v1(line: Line) -> bool:
    contain_formula = False
    # LibertineMathMI_italic_serifed_proportional
    # CMSY10_italic_serifed_proportional
    for span in line.spans:
        if span.block_type == "Text" and (
            "Math".lower() in span.font.lower() or "CMSY10".lower() in span.font.lower()
        ):
            contain_formula = True
            break
    return contain_formula


def if_contain_equation_v2(line: Line) -> bool:
    contain_formula = False
    size_set: set = set()
    flags_set: set = set()
    block_type: str = ""
    for span in line.spans:
        # condition1: math font
        if span.block_type == "Text" and (
            "Math".lower() in span.font.lower() or "CMSY10".lower() in span.font.lower()
        ):
            contain_formula = True
            break
        # condition2: diff size and diff flags
        size_set.add(span.size)
        flags_set.add(span.flags)
        block_type = span.block_type
    if block_type == "Text" and len(size_set) > 1 and len(flags_set) > 1:
        contain_formula = True
    return contain_formula


def if_contain_equation_v3(line: Line, spans_analyzer: SpansAnalyzer) -> bool:
    contain_formula = False
    for span in line.spans:
        if span.block_type == "Text" and (
            span.font != spans_analyzer.get_most_font_type(SpanType.Text)
        ):
            contain_formula = True
            break
    return contain_formula


def detect_inline_equations(
    page_idx: int,
    page: Page,
    debug_mode: bool = False,
):
    for block_idx, block in enumerate(page.blocks):
        for line_index, line in enumerate(block.lines):
            # check if line contain inline equation
            if if_contain_equation_v2(line):
                line_bboxes: List[List[float]] = []
                merged_line_bbox: List[float] = None

                # get prev & next line's bbox
                prev_bbox: List[float] = None
                next_bbox: List[float] = None
                if (line_index - 1) >= 0 and (line_index - 1) <= (len(block.lines) - 1):
                    prev_bbox = block.lines[line_index - 1].bbox
                if (line_index + 1) >= 0 and (line_index + 1) <= (len(block.lines) - 1):
                    next_bbox = block.lines[line_index + 1].bbox
                current_bbox = block.lines[line_index].bbox

                # resized line bbox
                x1 = current_bbox[0] - 5
                y1 = (
                    current_bbox[1]
                    if prev_bbox is None or current_bbox[1] > prev_bbox[3]
                    else (current_bbox[1] + ((prev_bbox[3] - current_bbox[1]) / 2)) + 1
                )
                x2 = current_bbox[2] + 5
                y2 = (
                    current_bbox[3]
                    if next_bbox is None or next_bbox[1] > current_bbox[3]
                    else (next_bbox[1] + ((current_bbox[3] - next_bbox[1]) / 2)) - 1
                )
                merged_line_bbox = [x1, y1, x2, y2]
                line_bboxes.append(merged_line_bbox)

                equation_image: io.BytesIO = get_image_bytes(page, merged_line_bbox)
                if equation_image is None:
                    continue


def detect_inline_equations_v2(
    page_idx: int,
    page: Page,
    debug_mode: bool = False,
):
    equations_info: list[EquationInfo] = []
    for block_idx, block in enumerate(page.blocks):
        containe_equations = False
        for line in block.lines:
            # check if line contain inline equation
            if if_contain_equation_v2(line):
                containe_equations = True
                break
        if not containe_equations:
            continue

        merged_bbox: List[float] = block.bbox

        # get block image
        equation_image: io.BytesIO = get_image_bytes(page, merged_bbox)
        if equation_image is None:
            continue

        if debug_mode:
            # Save equation image
            save_debug_info(equation_image, "inline_equations", page_idx, block_idx)
        equations_info.append(
            EquationInfo(
                content_base64=get_image_base64(equation_image),
                page_idx=page_idx,
                block_idx=block_idx,
            )
        )
    return equations_info
