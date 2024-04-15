from layoutana.bbox import merge_boxes
from layoutana.table import get_image_bytes
from layoutana.utils import convert_doc_to_pixel_coords, save_debug_info, merge_target_blocks, set_block_type
from layoutana.schema import Block, Page
from layoutana.settings import settings

import io



def merge_picture_blocks(pages: list[Page]):
    merge_target_blocks(pages, "Picture")


def extend_picture_blocks(pages: list[Page], debug_mode: bool):
    for page_idx, page in enumerate(pages):
        for block_idx, block in enumerate(page.blocks):
            if block.most_common_block_type() != "Picture":
                continue

            # merge picture blocks
            prev_block: Block = None
            next_block: Block = None
            if block_idx > 0:
                prev_block = page.blocks[block_idx - 1]
            if block_idx < len(page.blocks) - 1:
                next_block = page.blocks[block_idx + 1]

            prev_block_type = (
                prev_block.most_common_block_type() if prev_block else None
            )
            next_block_type = (
                next_block.most_common_block_type() if next_block else None
            )

            merged_bbox: list[float] = block.bbox

            if prev_block_type is not None and "Figure" in prev_block.prelim_text:
                # merge previous caption
                merged_bbox = merge_boxes(block.bbox, prev_block.bbox)
                set_block_type(prev_block, "Caption")
            elif next_block_type is not None and "Figure" in next_block.prelim_text:
                # merge next caption
                merged_bbox = merge_boxes(block.bbox, next_block.bbox)
                set_block_type(next_block, "Caption")
            else:
                set_block_type(block, "Picture")
                continue

            # get picture image
            picture_image: io.BytesIO = get_image_bytes(page, merged_bbox)
            if picture_image is None:
                continue

            # picture pixel bbox
            pixel_bbox = convert_doc_to_pixel_coords(merged_bbox, settings.NOUGAT_DPI)
            block.picture_pixel_bbox = pixel_bbox

            # save picture image
            if debug_mode:
                save_debug_info(picture_image, "picture", page_idx, block_idx)
