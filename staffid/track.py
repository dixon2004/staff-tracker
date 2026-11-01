import numpy as np
import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

OVERRIDES = dict(
    track_high_thresh=0.2, track_low_thresh=0.1, new_track_thresh=0.2,
    gmc_method="none",
)


class Tracker:

    def __init__(self) -> None:
        """Initialize the Tracker with BOTSORT."""
        self.tracker = BOTSORT(IterableSimpleNamespace(**{**YAML.load(check_yaml("botsort.yaml")), **OVERRIDES}))


    def update(self, boxes: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Update the tracker with new detections and return the track IDs.

        Args:
            boxes (np.ndarray): Array of shape (N, 5) containing bounding boxes in the format [x1, y1, x2, y2, conf].
            frame (np.ndarray): The current frame image.

        Returns:
            np.ndarray: Array of shape (N,) containing the track ID for each bounding box (-1 if untracked).
        """
        data = np.column_stack((boxes[:, :5], np.zeros(len(boxes)))).astype(np.float32)
        tracked = self.tracker.update(Boxes(torch.from_numpy(data), orig_shape=frame.shape[:2]).numpy(), frame)
        ids = np.full(len(boxes), -1)
        if len(tracked):
            ids[tracked[:, 7].astype(int)] = tracked[:, 4]
        return ids
