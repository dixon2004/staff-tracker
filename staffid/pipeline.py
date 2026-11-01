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


def _best_path(s: pd.DataFrame, weight: np.ndarray, max_speed: float) -> pd.DataFrame:
    """
    Keep the chain of candidates that one person could really walk, with the highest total weight.

    Args:
        s (pd.DataFrame): Candidate boxes sorted by frame, with x and y centre columns.
        weight (np.ndarray): How much each candidate is worth keeping.
        max_speed (float): Furthest a person can move between frames, in pixels.

    Returns:
        pd.DataFrame: The candidates on the best chain, at most one per frame.
    """
    if s.empty:
        return s
    f, x, y = s.frame.to_numpy(), s.x.to_numpy(), s.y.to_numpy()
    best = weight.copy()
    prev = np.full(len(s), -1)
    for i in range(len(s)):
        j = np.flatnonzero((f[:i] < f[i]) & (np.hypot(x[:i] - x[i], y[:i] - y[i]) <= max_speed * (f[i] - f[:i])))
        if len(j):
            prev[i] = j[best[j].argmax()]
            best[i] += best[prev[i]]
    keep, i = [], int(best.argmax())
    while i >= 0:
        keep.append(i)
        i = prev[i]
    return s.iloc[keep[::-1]]


def _moving_boxes(dets: pd.DataFrame, max_gap: int, max_speed: float) -> pd.DataFrame:
    """
    Drop boxes that stay in one place, such as seated people, and small boxes that only cover part of a person.

    Args:
        dets (pd.DataFrame): All person boxes in frame order, with x and y centre columns.
        max_gap (int): How many frames to look before and after each box.
        max_speed (float): Furthest a person can move between frames, in pixels.

    Returns:
        pd.DataFrame: The large enough boxes with few other boxes at the same spot nearby in time.
    """
    f, x, y = dets.frame.to_numpy(), dets.x.to_numpy(), dets.y.to_numpy()
    lo, hi = np.searchsorted(f, f - max_gap), np.searchsorted(f, f + max_gap, side="right")
    near = [(np.hypot(x[a:b] - x[i], y[a:b] - y[i]) < max_speed / 2).sum() for i, (a, b) in enumerate(zip(lo, hi))]
    return dets[(np.array(near, int) < max_gap) & (np.hypot(dets.x2 - dets.x1, dets.y2 - dets.y1) > 1.5 * max_speed)]


def _fill_gaps(s: pd.DataFrame, dets: pd.DataFrame, max_gap: int, max_speed: float) -> pd.DataFrame:
    """
    Fill short gaps in the path with real boxes of moving people, so the box follows the person.

    These boxes have no tag evidence, so they must fit a stricter speed limit than the path itself.

    Args:
        s (pd.DataFrame): The staff path, sorted by frame.
        dets (pd.DataFrame): All person boxes in frame order, with x and y centre columns.
        max_gap (int): Longest gap to fill, in frames.
        max_speed (float): Furthest a person can move between frames, in pixels.

    Returns:
        pd.DataFrame: The path with the extra boxes added, sorted by frame.
    """
    extra = _moving_boxes(dets, max_gap, max_speed)
    filled = [s]
    f = s.frame.to_numpy()
    for i in np.flatnonzero((np.diff(f) > 1) & (np.diff(f) <= max_gap + 1)):
        seg = pd.concat([s.iloc[[i]], extra[extra.frame.between(f[i] + 1, f[i + 1] - 1)], s.iloc[[i + 1]]])
        path = _best_path(seg, 1 + 1000 * seg.index.isin(s.index), max_speed * 0.65)
        if path.frame.iloc[0] == f[i] and path.frame.iloc[-1] == f[i + 1]:
            filled.append(path.iloc[1:-1])
    return pd.concat(filled).sort_values("frame")


def identify_staff(
        dets: pd.DataFrame, n_frames: int, tag_thresh: float = 0.2,
        reach: int = 100, min_hits: int = 2, strong: float = 0.3, max_gap: int = 50, max_speed: float = 0.45
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
        max_gap (int): Longest gap of missing frames to fill.
        max_speed (float): Furthest the staff member can move between frames, as a share of a typical person box diagonal.

    Returns:
        pd.DataFrame: A DataFrame containing the identified staff member's bounding boxes and tag information for each frame, with short gaps filled.
    """
    dets = dets.assign(x=(dets.x1 + dets.x2) / 2, y=(dets.y1 + dets.y2) / 2)
    d = dets[dets.tid >= 0].assign(gap=np.inf)
    for tid, hits in d[d.tag_conf >= tag_thresh].groupby("tid"):
        if len(hits) >= min_hits and hits.tag_conf.max() >= strong:
            t = d[d.tid == tid]
            d.loc[t.index, "gap"] = np.abs(t.frame.to_numpy()[:, None] - hits.frame.to_numpy()[None, :]).min(1)

    s = d[d.gap <= reach].sort_values(["frame", "gap", "tag_conf"], ascending=[True, True, False])
    limit = max_speed * np.hypot(d.x2 - d.x1, d.y2 - d.y1).median()
    s = _best_path(s, 1 + 1000 * (s.tag_conf.to_numpy() >= tag_thresh), limit)
    s = _fill_gaps(s, dets, max_gap, limit)
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
