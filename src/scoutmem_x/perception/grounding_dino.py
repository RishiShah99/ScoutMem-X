"""Grounding DINO perception adapter for open-vocabulary object detection."""

from __future__ import annotations

import logging
from typing import Any

from scoutmem_x.perception.adapters import Detection

logger = logging.getLogger(__name__)


class GroundingDINOAdapter:
    """PerceptionAdapter backed by Grounding DINO (HuggingFace transformers).

    Requires: ``pip install transformers torch pillow``

    Uses ``IDEA-Research/grounding-dino-tiny`` by default.  The model is
    loaded lazily on first ``predict`` call so import cost stays low.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str | None = None,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ) -> None:
        self.model_id = model_id
        self._device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._model: Any = None
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "GroundingDINOAdapter requires transformers and torch. "
                "Install with: pip install transformers torch"
            ) from exc

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading %s on %s …", self.model_id, device)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(device)
        self._device = device

    def predict(self, observation: Any, query: str) -> list[Detection]:
        rgb_path: str | None = getattr(observation, "rgb_path", None)
        if rgb_path is None:
            return []

        self._ensure_loaded()

        import torch
        from PIL import Image as PILImage

        image = PILImage.open(rgb_path).convert("RGB")
        inputs = self._processor(
            images=image, text=query, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        metadata_base = getattr(observation, "metadata", {})
        room_name = metadata_base.get("room_name", "detected_region")

        detections: list[Detection] = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"], strict=False
        ):
            x1, y1, x2, y2 = (int(v) for v in box.tolist())
            detections.append(
                Detection(
                    label=label,
                    score=float(score),
                    region=(x1, y1, x2, y2),
                    metadata={
                        "query": query,
                        "source": "grounding_dino",
                        "region": room_name,
                        "target_label": query.replace("find the ", ""),
                    },
                )
            )
        return detections
