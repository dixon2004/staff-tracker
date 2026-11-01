import numpy as np
from ultralytics import YOLO

from . import MODELS


class TagDetector:

    def __init__(
            self, weights: str = str(MODELS / "tag_yolo26n.pt"),
            conf: float = 0.05, imgsz: int = 1280, device: str = None, margin: int = 8
            ) -> None:
        """
        Initialize the TagDetector.

        Args:
            weights (str): Path to the YOLO model weights.
            conf (float): Confidence threshold for detection.
            imgsz (int): Size of the input image.
            device (str): Device to run the model on.
            margin (int): Margin around the person box to consider for tag detection.
        """
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.device, self.margin = conf, imgsz, device, margin


    def __call__(self, frame: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run tag detection on the input frame and associate detected tags with person boxes.

        Args:
            frame (np.ndarray): Input frame.
            boxes (np.ndarray): Array of shape (N, 5) containing person bounding boxes in the format [x1, y1, x2, y2, conf].

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - out_conf (np.ndarray): Array of shape (N,) containing the confidence of the tag associated with each person box (0 if none).
                - out_xy (np.ndarray): Array of shape (N, 2) containing the (x, y) coordinates of the detected tags associated with each person box. If no tag is detected for a person box, the coordinates will be NaN.
        """
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0].boxes
        out_conf, out_xy = np.zeros(len(boxes)), np.full((len(boxes), 2), np.nan)
        x1, y1, x2, y2 = boxes[:, :4].T
        m = self.margin
        for c, (x, y) in zip(r.conf.cpu().numpy(), r.xywh.cpu().numpy()[:, :2]):
            inside = np.flatnonzero((x1 - m <= x) & (x <= x2 + m) & (y1 - m <= y) & (y <= y2 + m))
            if len(inside) == 0:
                continue
            i = inside[np.hypot((x1 + x2)[inside] / 2 - x, (y1 + y2)[inside] / 2 - y).argmin()]
            if c > out_conf[i]:
                out_conf[i], out_xy[i] = c, (x, y)
        return out_conf, out_xy
