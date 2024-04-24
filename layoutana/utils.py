import logging
from layoutana.bbox import merge_boxes
from layoutana.schema import Block, Page
from layoutana.settings import settings

from typing import Callable, List, TypeVar
from copy import deepcopy
from PIL import Image

import io
import os
import base64


def convert_doc_to_pixel_coords(doc_bbox, dpi):
    """Convert document coordinates to pixel coordinates based on DPI."""
    x1, y1, x2, y2 = doc_bbox
    scale_factor = dpi / 72
    px1 = int(x1 * scale_factor)
    py1 = int(y1 * scale_factor)
    px2 = int(x2 * scale_factor)
    py2 = int(y2 * scale_factor)
    return (px1, py1, px2, py2)


def get_page_image(inner_page: Page, crop_bbox: list[float] = None):
    image_byte = base64.b64decode(inner_page.image_info.content_base64)
    buffered = io.BytesIO(image_byte)
    image = Image.open(buffered)
    if crop_bbox is not None:
        image = image.crop(convert_doc_to_pixel_coords(crop_bbox, settings.NOUGAT_DPI))
    return (
        image,
        inner_page.image_info.pt_bbox,
        inner_page.image_info.pt_width,
        inner_page.image_info.pt_height,
    )


def get_image_bytes(page: Page, merged_block_bbox):
    try:
        png_image, _, _, _ = get_page_image(page, merged_block_bbox)
        png_image = png_image.convert("RGB")

        img_out = io.BytesIO()
        png_image.save(img_out, format="PNG")
        if img_out is None:
            return None
        return img_out

    except Exception as exception:
        logging.error(exception)
        return None


def get_image_base64(image_bytesIo: io.BytesIO):
    return base64.b64encode(image_bytesIo.getvalue()).decode("utf-8")


def set_block_type(block: Block, type: str):
    for line in block.lines:
        for span in line.spans:
            span.block_type = type


def set_special_block_type(block: Block, origin_type: str, type: str):
    for line in block.lines:
        for span in line.spans:
            if span.block_type == origin_type:
                span.block_type = type


def merge_target_blocks(pages: List[Page], block_type: str):
    target_lines = []
    target_bbox = None
    for page in pages:
        new_page_blocks = []
        for block in page.blocks:
            # other block
            if block.most_common_block_type() != block_type:
                if len(target_lines) > 0:
                    # merge last target block
                    target_block = Block(
                        lines=deepcopy(target_lines), pnum=page.pnum, bbox=target_bbox
                    )
                    new_page_blocks.append(target_block)
                    # clear target block
                    target_lines = []
                    target_bbox = None

                # merge other block
                new_page_blocks.append(block)
                continue

            # merge target block
            target_lines.extend(block.lines)
            if target_bbox is None:
                # init target bbox
                target_bbox = block.bbox
            else:
                # merge target bbox
                target_bbox = merge_boxes(target_bbox, block.bbox)

        if len(target_lines) > 0:
            # merge last target block
            target_block = Block(
                lines=deepcopy(target_lines), pnum=page.pnum, bbox=target_bbox
            )
            new_page_blocks.append(target_block)
            # clear target block
            target_lines = []
            target_bbox = None

        # update new page blocks
        page.blocks = new_page_blocks


T = TypeVar("T")


def save_debug_doc_info(
    type: str, items: list[T], processor: Callable[[T], str]
) -> list[str]:
    real_path = f"debug/doc_info/{type}.txt"
    os.makedirs(os.path.dirname(real_path), exist_ok=True)
    with open(real_path, "w", encoding="utf-8") as file:
        for item in items:
            file.write(processor(item) + "\n")


def save_debug_info(image, model_name, page_idx, block_idx=0, results=None):
    # create file path
    real_path = f"debug/{model_name}/page_{page_idx}_{block_idx}.png"
    os.makedirs(os.path.dirname(real_path), exist_ok=True)

    # save input image
    if isinstance(image, io.BytesIO):
        with open(real_path, "wb") as f:
            f.write(image.getvalue())
    else:
        image.save(real_path)

    # save output result
    if results is not None:
        image_link = (
            f"![page_{page_idx}_{block_idx}.png](page_{page_idx}_{block_idx}.png)"
        )

        with open(f"debug/{model_name}/result.md", "a") as file:
            file.write(f"source_{page_idx}_{block_idx}: \n\n")
            file.write(image_link + " \n\n")
            file.write(f"result_{page_idx}_{block_idx}: \n\n")
            for result in results:
                file.write(f"{result}  \n\n")
