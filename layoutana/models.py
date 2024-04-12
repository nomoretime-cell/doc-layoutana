from layoutana.settings import settings
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3Processor,
)


class ModelInfo:
    def __init__(self, processor, model):
        self._processor = processor
        self._model = model

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model


def load_segment_model():
    processor = LayoutLMv3Processor.from_pretrained(
        settings.LAYOUT_MODEL_NAME, apply_ocr=False
    )

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        settings.LAYOUT_MODEL_NAME,
        torch_dtype=settings.MODEL_DTYPE,
    ).to(settings.TORCH_DEVICE)

    model.config.id2label = {
        0: "Caption",
        1: "Footnote",
        2: "Formula",
        3: "List-item",
        4: "Page-footer",
        5: "Page-header",
        6: "Picture",
        7: "Section-header",
        8: "Table",
        9: "Text",
        10: "Title",
    }

    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    return ModelInfo(processor, model)


def load_ordering_model():
    processor = LayoutLMv3Processor.from_pretrained(settings.ORDERER_MODEL_NAME)
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        settings.ORDERER_MODEL_NAME,
        torch_dtype=settings.MODEL_DTYPE,
    ).to(settings.TORCH_DEVICE)
    model.eval()
    return ModelInfo(processor, model)
