from layoutana.settings import settings
from layoutana.headers import filter_header_footer
from layoutana.ordering import order_blocks
from layoutana.schema import BlockType, Page, Span
from layoutana.segmentation import get_pages_types
from layoutana.spans import SpanType, SpansAnalyzer
from layoutana.utils import save_debug_doc_info
from layoutana.models import ModelInfo, load_ordering_model, load_segment_model

import logging
import os
import threading

from pyfunvice import faas, start_faas, start_fass_with_uvicorn, start_fass_with_cmd

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
    return pages


@faas(path="/api/v1/parser/ppl/layout", inparam_type="flat")
async def process(pages: list[Page]):
    logging.info(
        f"POST request, pid: {os.getpid()}, thread id: {threading.current_thread().ident}"
    )
    pages_instance: list[Page] = []
    for page in pages:
        pages_instance.append(Page(**page))
    pages = inner_process(pages=pages_instance, debug_mode=settings.DEBUG)
    return {"pages": pages}


if __name__ == "__main__":
    start_faas(workers=settings.WORKER_NUM, port=8001, post_fork_func=post_fork_func)
