from layoutana.settings import settings
from layoutana.headers import filter_header_footer
from layoutana.ordering import order_blocks
from layoutana.schema import (
    BlockImage,
    BlockType,
    EquationInfo,
    ImageInfo,
    Page,
    PictureInfo,
    Span,
    TableInfo,
)
from layoutana.segmentation import get_pages_types
from layoutana.spans import SpanType, SpansAnalyzer
from layoutana.table import detect_tables, recognition_table
from layoutana.equations import detect_equations
from layoutana.picture import detect_pictures
from layoutana.utils import (
    get_image_base64,
    get_image_bytes,
    merge_target_blocks,
    save_debug_doc_info,
    save_debug_info,
)
from layoutana.models import ModelInfo, load_ordering_model, load_segment_model

import logging
import os
import threading

from pyfunvice import (
    app_service,
    start_app,
    get_app_instance,
)

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(thread)d] [%(levelname)s] %(message)s"
)

model_lst: list[ModelInfo] = []


def post_fork_func():
    global model_lst
    model_lst = [load_segment_model(), load_ordering_model()]


def annotate_spans_type(pages: list[Page], pages_types: list[list[BlockType]]):
    for i, page in enumerate(pages):
        page_types = pages_types[i]
        page.add_types(page_types)


def get_all_spans(pages: list[Page]) -> list[Span]:
    spans: list[Span] = []
    for page in pages:
        for block in page.blocks:
            for line in block.lines:
                for span in line.spans:
                    spans.append(span)
    # FOR DEBUG
    save_debug_doc_info("spans_type", spans, lambda x: str(x))
    return spans


def inner_process(
    pages: list[Page],
    parallel_factor: int = 1,
    debug_mode: bool = False,
):
    # Unpack models from list
    segment_model_info, order_model_info = model_lst
    out_meta: dict = {}
    pages_types: list[list[BlockType]] = get_pages_types(
        segment_model_info,
        pages,
        batch_size=settings.LAYOUT_BATCH_SIZE * parallel_factor,
        debug_mode=debug_mode,
    )

    # Find headers and footers
    bad_span_ids: list[int] = filter_header_footer(pages)
    out_meta["block_stats"] = {"header_footer": len(bad_span_ids)}

    annotate_spans_type(pages, pages_types)

    # Get text font size
    spans: list[Span] = get_all_spans(pages)
    spans_analyzer = SpansAnalyzer(spans)
    if len(spans_analyzer.type2fontSize[SpanType.Text.value]) > 0:
        for page in pages:
            page.text_font = spans_analyzer.type2fontSize[SpanType.Text.value][
                0
            ].font_size
    # update_equations_in_spans(pages, pages_types)

    pages = order_blocks(
        order_model_info,
        pages,
        batch_size=settings.ORDERER_BATCH_SIZE * parallel_factor,
    )

    block_image: BlockImage = BlockImage()

    # # Detect tables
    # tables_info = detect_tables(pages, debug_mode)
    # # Detect pictures
    # pictures_info = detect_pictures(pages, debug_mode)
    # # Detect equations
    # equations_info = detect_equations(pages, debug_mode)

    tables_info, pictures_info, equations_info, type_block_num = detect_all(pages)

    block_image.tables_info = tables_info
    block_image.pictures_info = pictures_info
    block_image.equations_info = equations_info

    pages[0].type_block_idx = type_block_num - 1
    pages[0].type_block_num = type_block_num
    pages[0].block_idx = len(pages[0].blocks)
    return pages, block_image


global_page_idx = 0


def detect_all(pages):
    global global_page_idx
    tables_info: list[TableInfo] = []
    pictures_info: list[PictureInfo] = []
    equations_info: list[EquationInfo] = []

    merge_target_blocks(pages, "Table")
    merge_target_blocks(pages, "Picture")
    merge_target_blocks(pages, "Formula")

    for page_idx, page in enumerate(pages):
        global_page_idx = global_page_idx + 1
        for block_idx, block in enumerate(page.blocks):
            image_bytes = get_image_bytes(pages[page_idx], block.bbox)
            if image_bytes is None:
                continue
            if block.most_common_block_type() == "Table":
                table_text = recognition_table(block, block_idx, False)
                tables_info.append(
                    TableInfo(
                        type="table",
                        content_base64=get_image_base64(image_bytes),
                        block_idx=block_idx,
                        text=table_text,
                    )
                )
                save_debug_info(
                    image_bytes,
                    "table",
                    global_page_idx,
                    block_idx,
                    [table_text, "predictions"],
                )
            elif block.most_common_block_type() == "Picture":
                pictures_info.append(
                    PictureInfo(
                        type="picture",
                        content_base64=get_image_base64(image_bytes),
                        block_idx=block_idx,
                    )
                )
                save_debug_info(image_bytes, "picture", global_page_idx, block_idx)
            elif block.most_common_block_type() == "Formula":
                equations_info.append(
                    EquationInfo(
                        type="equation",
                        content_base64=get_image_base64(image_bytes),
                        block_idx=block_idx,
                    )
                )
                save_debug_info(image_bytes, "equations", global_page_idx, block_idx)

    type_block_idx = 0
    type_block_num = len(tables_info) + len(pictures_info) + len(equations_info) + 1
    for _, table in enumerate(tables_info):
        table.type_block_idx = type_block_idx
        type_block_idx = type_block_idx + 1
        table.type_block_num = type_block_num
    for _, picture in enumerate(pictures_info):
        picture.type_block_idx = type_block_idx
        type_block_idx = type_block_idx + 1
        picture.type_block_num = type_block_num
    for _, equation in enumerate(equations_info):
        equation.type_block_idx = type_block_idx
        type_block_idx = type_block_idx + 1
        equation.type_block_num = type_block_num
    return tables_info, pictures_info, equations_info, type_block_num


@app_service(path="/api/v1/parser/ppl/layout", inparam_type="flat")
async def process(image_info: ImageInfo, page_info: Page):
    logging.info(
        f"POST request, pid: {os.getpid()}, thread id: {threading.current_thread().ident}"
    )
    pages_instance: list[Page] = []

    page_info["image_info"] = image_info
    pages_instance.append(Page(**page_info))
    pages, block_image = inner_process(pages=pages_instance, debug_mode=settings.DEBUG)

    result_array = []
    result_array.extend(block_image.tables_info)
    result_array.extend(block_image.pictures_info)
    result_array.extend(block_image.equations_info)
    result_array.extend(pages)
    return result_array


if __name__ == "__main__":
    start_app(workers=settings.WORKER_NUM, port=8001, post_fork_func=post_fork_func)

# app = get_app_instance(post_fork_func)
# poetry run gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001
