import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

KNOWN_SIZES_M = {
    "car": 1.5,
    "truck": 3.0,
    "bus": 3.2,
}

COCO_LABEL_MAP = {3: "car", 6: "bus", 8: "truck"}


class ObjectScaleRecovery:
    def __init__(self, device: str = None, score_thresh: float = 0.6):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights).to(self.device).eval()
        self.transform = weights.transforms()
        self.score_thresh = score_thresh

    @torch.no_grad()
    def detect(self, image_np: np.ndarray):
        tensor = self.transform(torch.from_numpy(image_np).permute(2, 0, 1)).to(self.device)
        output = self.model([tensor])[0]

        detections = []
        for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
            if score < self.score_thresh:
                continue
            cls = COCO_LABEL_MAP.get(int(label))
            if cls is None:
                continue
            x1, y1, x2, y2 = box.cpu().numpy()
            detections.append({"class": cls, "box": (x1, y1, x2, y2), "score": float(score)})
        return detections


def calibrate_object_based(relative_depth: np.ndarray, image_np: np.ndarray,
                            gsd_estimate_m_per_px: float = None):
    detector = ObjectScaleRecovery()
    detections = detector.detect(image_np)

    if not detections:
        return None, {"tier": "B_object_scale", "status": "no_objects_found"}

    scale_estimates = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        pixel_height = y2 - y1
        real_height = KNOWN_SIZES_M[det["class"]]
        scale_estimates.append(real_height / pixel_height)

    m_per_px = float(np.median(scale_estimates))
    approx_max_height_m = m_per_px * relative_depth.shape[0] * 0.1
    metric_dsm = relative_depth * approx_max_height_m

    return metric_dsm, {
        "tier": "B_object_scale",
        "objects_used": len(detections),
        "m_per_px_estimate": m_per_px,
    }
