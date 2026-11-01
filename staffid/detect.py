import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

from . import MODELS

ROTATIONS = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]


def pick_device() -> str:
    """
    Pick the best available device for PyTorch.

    Returns:
        str: The device string to use with PyTorch.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _unrotate(xyxy: np.ndarray, k: int, w: int, h: int) -> np.ndarray:
    """
    Map boxes predicted on rotation k back to the original frame.

    Args:
        xyxy (np.ndarray): Array of shape (N, 4) containing bounding boxes in the format [x1, y1, x2, y2].
        k (int): The rotation index (0: no rotation, 1: 90° clockwise, 2: 180°, 3: 90° counterclockwise).
        w (int): Width of the original frame.
        h (int): Height of the original frame.

    Returns:
        np.ndarray: Array of shape (N, 4) containing the unrotated bounding boxes in the format [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = xyxy.T
    if k == 0:
        pts = [(x1, y1), (x2, y2)]
    elif k == 1:
        pts = [(y1, h - 1 - x1), (y2, h - 1 - x2)]
    elif k == 2:
        pts = [(w - 1 - x1, h - 1 - y1), (w - 1 - x2, h - 1 - y2)]
    else:
        pts = [(w - 1 - y1, x1), (w - 1 - y2, x2)]
    (ax, ay), (bx, by) = pts
    return np.stack([np.minimum(ax, bx), np.minimum(ay, by), np.maximum(ax, bx), np.maximum(ay, by)], 1)


class PersonDetector:

    def __init__(
            self, weights: str=MODELS / "yolo26n.pt", 
            conf: float=0.2, imgsz: int=960, device: str=None
            ) -> None:
        """
        Initialize the PersonDetector.

        Args:
            weights (Path): Path to the model weights.
            conf (float): Confidence threshold for detections.
            imgsz (int): Size of the input image.
            device (str): Device to use for inference.
        """
        self.model = YOLO(str(weights))
        self.conf, self.imgsz = conf, imgsz
        self.device = device or pick_device()


    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """
        Run person detection on the input frame.

        Args:
            frame (np.ndarray): Input frame.

        Returns:
            np.ndarray: Array of shape (N, 5) containing the detected person bounding boxes in the format [x1, y1, x2, y2, conf].
        """
        h, w = frame.shape[:2]
        views = [frame if r is None else cv2.rotate(frame, r) for r in ROTATIONS]
        results = self.model.predict(views, classes=[0], conf=self.conf, imgsz=self.imgsz,
                                     device=self.device, verbose=False)
        boxes, scores = [], []
        for k, r in enumerate(results):
            if len(r.boxes):
                boxes.append(_unrotate(r.boxes.xyxy.cpu().numpy(), k, w, h))
                scores.append(r.boxes.conf.cpu().numpy())
        if not boxes:
            return np.zeros((0, 5), np.float32)
        b, s = np.concatenate(boxes), np.concatenate(scores)
        keep = torchvision.ops.nms(torch.from_numpy(b).float(), torch.from_numpy(s).float(), 0.45).numpy()
        out = np.c_[b[keep], s[keep]]
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, w - 1)
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, h - 1)
        return out.astype(np.float32)
