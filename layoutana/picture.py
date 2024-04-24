from layoutana.bbox import merge_boxes
from layoutana.utils import (
    convert_doc_to_pixel_coords,
    get_image_base64,
    get_image_bytes,
    save_debug_info,
    merge_target_blocks,
    set_block_type,
)
from layoutana.schema import Block, Page, PictureInfo
from layoutana.settings import settings

import io


def detect_pictures(pages: list[Page], debug_mode: bool):
    merge_target_blocks(pages, "Picture")
    pictures_info: list[PictureInfo] = []
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
            image_bytes: io.BytesIO = get_image_bytes(page, merged_bbox)
            if image_bytes is None:
                continue

            # picture pixel bbox
            pixel_bbox = convert_doc_to_pixel_coords(merged_bbox, settings.NOUGAT_DPI)
            block.picture_pixel_bbox = pixel_bbox

            # save picture image
            if debug_mode:
                save_debug_info(image_bytes, "picture", page_idx, block_idx)
            pictures_info.append(
                PictureInfo(
                    content_base64=get_image_base64(image_bytes),
                    page_idx=page_idx,
                    block_idx=block_idx,
                )
            )
    return pictures_info
