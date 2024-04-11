from layoutana.headers import filter_header_footer
from layoutana.ordering import load_ordering_model, order_blocks
from layoutana.schema import BlockType, Page, Span
from layoutana.segmentation import get_pages_types, load_segment_model
from layoutana.settings import settings
from layoutana.spans import SpanType, SpansAnalyzer
from layoutana.utils import save_debug_doc_info

from pyfunvice import faas, start_faas, start_fass_with_uvicorn, start_fass_with_cmd

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
    segment_model, order_model = model_lst
    out_meta: dict = {}
    pages_types: list[list[BlockType]] = get_pages_types(
        segment_model,
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
        pages,
        order_model,
        batch_size=settings.ORDERER_BATCH_SIZE * parallel_factor,
    )
    return pages


@faas(path="/api/v1/parser/ppl/layout", inparam_type="flat")
async def process(pages: list[Page]):
    pages_instance: list[Page] = []
    for page in pages:
        pages_instance.append(Page(**page))
    pages = inner_process(pages_instance, settings.DEBUG)
    return {"pages": pages}


if __name__ == "__main__":
    start_fass_with_uvicorn(workers=settings.WORKER_NUM, port=8001)
