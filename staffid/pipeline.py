import hashlib
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import CACHE, MODELS
from .detect import PersonDetector
from .tag import TagDetector
from .track import Tracker


def video_info(path: Path) -> dict:
    """
    Get basic information about a video file.

    Args:
        path (Path): Path to the video file.

    Returns:
        dict: A dictionary containing the number of frames, frames per second (fps), width, and height of the video.
    """
    cap = cv2.VideoCapture(str(path))
    info = dict(frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def iter_frames(path: Path) -> Iterator[tuple[int, np.ndarray]]:
    """
    Iterate over frames in a video file.

    Args:
        path (Path): Path to the video file.

    Yields:
        tuple: A tuple containing the frame index and the frame image (as a NumPy array in BGR format).
    """
    cap = cv2.VideoCapture(str(path))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame
        i += 1
    cap.release()


def cache_key(video: Path) -> str:
    """
    Generate a cache key for a video file based on its content and the current code state.

    Args:
        video (Path): Path to the video file.

    Returns:
        str: A 12-character hexadecimal string representing the cache key.
    """
    h = hashlib.md5(video.read_bytes())
    for f in (MODELS / "tag_yolo26n.pt", *sorted(Path(__file__).parent.glob("*.py"))):
        st = os.stat(f)
        h.update(f"{st.st_size}:{st.st_mtime}".encode())
    return h.hexdigest()[:12]


def detect_and_score(video: Path, progress=None) -> pd.DataFrame:
    """
    Run person detection, tracking and tag detection on every frame. Results are cached.

    Args:
        video (Path): Path to the video file.
        progress (callable, optional): A function to report progress. It should accept two arguments: the number of frames done and the total number of frames.

    Returns:
        pd.DataFrame: A DataFrame containing detection results for each frame, including bounding boxes, track IDs, confidence scores, and tag information.
    """
    cache = CACHE / f"{cache_key(video)}_dets.csv"
    if cache.exists():
        return pd.read_csv(cache)
    detector = PersonDetector()
    tagger = TagDetector(device=detector.device)
    tracker = Tracker()
    n = video_info(video)["frames"]
    rows = []
    for i, frame in iter_frames(video):
        boxes = detector(frame)
        ids = tracker.update(boxes, frame)
        tag_conf, tag_xy = tagger(frame, boxes)
        rows += [(i, t, *b, c, tx, ty) for t, b, c, (tx, ty) in zip(ids, boxes, tag_conf, tag_xy)]
        if progress:
            progress(i + 1, n)
    df = pd.DataFrame(rows, columns=["frame", "tid", "x1", "y1", "x2", "y2", "conf", "tag_conf", "tag_x", "tag_y"])
    df.to_csv(cache, index=False)
    return df


def identify_staff(
        dets: pd.DataFrame, n_frames: int, tag_thresh: float = 0.2,
        reach: int = 100, min_hits: int = 2, strong: float = 0.5, max_gap: int = 12
        ) -> pd.DataFrame:
    """
    Identify the staff member from the detection results.

    Args:
        dets (pd.DataFrame): DataFrame containing detection results for each frame.
        n_frames (int): Total number of frames in the video.
        tag_thresh (float): Confidence threshold for tag detection.
        reach (int): How many frames before and after a sighting the staff label stays on a track.
        min_hits (int): Minimum number of frames with tag detection to consider a track as staff.
        strong (float): Confidence threshold for strong tag detection.
        max_gap (int): Maximum gap of missing frames to interpolate.

    Returns:
        pd.DataFrame: A DataFrame containing the identified staff member's bounding boxes and tag information for each frame, with short gaps interpolated.
    """
    d = dets[dets.tid >= 0].assign(gap=np.inf)
    for tid, hits in d[d.tag_conf >= tag_thresh].groupby("tid"):
        if len(hits) >= min_hits and hits.tag_conf.max() >= strong:
            t = d[d.tid == tid]
            d.loc[t.index, "gap"] = np.abs(t.frame.to_numpy()[:, None] - hits.frame.to_numpy()[None, :]).min(1)

    s = d[d.gap <= reach]
    s = s.sort_values(["gap", "tag_conf"], ascending=[True, False]).drop_duplicates("frame")
    s = s.assign(x=(s.x1 + s.x2) / 2, y=(s.y1 + s.y2) / 2)
    cols = ["x", "y", "x1", "y1", "x2", "y2"]
    per_frame = pd.DataFrame({"frame": np.arange(n_frames)}).merge(
        s[["frame", "tid", *cols, "tag_conf", "tag_x", "tag_y"]], on="frame", how="left"
    ).astype({c: float for c in [*cols, "tag_conf", "tag_x", "tag_y"]})

    missing = per_frame.x.isna()
    gap_len = missing.groupby((~missing).cumsum()).transform("sum")
    fill = missing & (gap_len <= max_gap)
    per_frame.loc[fill, cols] = per_frame[cols].interpolate(limit_area="inside").loc[fill]

    per_frame["staff_present"] = per_frame.x.notna()
    per_frame["tag_visible"] = per_frame.tag_conf >= tag_thresh
    return per_frame


def frame_ranges(per_frame: pd.DataFrame) -> list[tuple[int, int]]:
    """
    Get the ranges of frames where the staff member is present.

    Args:
        per_frame (pd.DataFrame): DataFrame containing the identified staff member's bounding boxes and tag information for each frame.

    Returns:
        list[tuple[int, int]]: A list of tuples representing the start and end frame indices where the staff member is present.
    """
    f = per_frame.frame[per_frame.staff_present].to_numpy()
    if len(f) == 0:
        return []
    breaks = np.where(np.diff(f) > 1)[0]
    return list(zip(np.r_[f[0], f[breaks + 1]].tolist(), np.r_[f[breaks], f[-1]].tolist()))


def render_video(video: Path, per_frame: pd.DataFrame, out_path: Path, progress=None) -> None:
    """
    Render a video with the identified staff member annotated.

    Args:
        video (Path): Path to the input video file.
        per_frame (pd.DataFrame): DataFrame containing the identified staff member's bounding boxes and tag information for each frame.
        out_path (Path): Path to save the output annotated video.
        progress (callable, optional): A function to report progress. It should accept two arguments: the number of frames done and the total number of frames.
    """
    info = video_info(video)
    tmp = f"{out_path}.tmp.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (info["width"], info["height"]))
    staff = per_frame.set_index("frame")
    for i, frame in iter_frames(video):
        if i in staff.index and staff.at[i, "staff_present"]:
            r = staff.loc[i]
            cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 220, 0), 3)
            cv2.putText(frame, "STAFF", (int(r.x1), max(15, int(r.y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 220, 0), 2)
            if r.tag_visible:
                cv2.circle(frame, (int(r.tag_x), int(r.tag_y)), 7, (0, 255, 255), 2)
        cv2.putText(frame, f"frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.write(frame)
        if progress:
            progress(i + 1, info["frames"])
    writer.release()
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(out_path)], check=True)
        os.remove(tmp)
    except (OSError, subprocess.CalledProcessError):
        os.replace(tmp, out_path)
