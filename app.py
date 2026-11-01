import os

import altair as alt
import pandas as pd
import streamlit as st

from staffid import CACHE
from staffid.pipeline import cache_key, detect_and_score, frame_ranges, identify_staff, render_video, video_info

st.set_page_config(page_title="Staff Finder", layout="wide")
st.title("Staff identification")
st.caption("Finds the person wearing the staff name tag in a FootfallCam top-down video.")

CACHE.mkdir(exist_ok=True)
with st.container(border=True):
    upload = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if upload is None:
        st.stop()
    video = CACHE / f"upload{os.path.splitext(upload.name)[1]}"
    video.write_bytes(upload.getvalue())
    info = video_info(video)
    if info["frames"] <= 0:
        st.error("This video could not be read.")
        st.stop()
    st.caption(f"🎞️ **{upload.name}** · {info['frames']} frames · {info['frames'] / info['fps']:.1f} s · "
               f"{info['fps']:.0f} fps · {info['width']}×{info['height']}")


def show_progress(bar, label):
    return lambda i, n: bar.progress(min(i / n, 1.0), text=f"{label} {i}/{n}")


key = cache_key(video)
for f in CACHE.iterdir():
    if f.is_file() and not f.samefile(video) and not f.name.startswith(key):
        f.unlink()

bar = st.progress(0.0, text="Detecting people and tags…")
dets = detect_and_score(video, progress=show_progress(bar, "Processing frame"))
bar.empty()
per_frame = identify_staff(dets, info["frames"])
present = per_frame[per_frame.staff_present]

c1, c2, c3 = st.columns(3)
c1.metric("Frames with staff", f"{len(present)} / {info['frames']}")
c2.metric("Share of clip", f"{len(present) / info['frames']:.0%}")
c3.metric("Frames with tag visible", int(per_frame.tag_visible.sum()))

st.subheader("Task 1: frames with staff present")
ranges = frame_ranges(per_frame)
if ranges:
    fps = info["fps"]
    st.dataframe(pd.DataFrame([{"start frame": a, "end frame": b, "frames": b - a + 1,
                                "start (s)": round(a / fps, 2), "end (s)": round(b / fps, 2)} for a, b in ranges]),
                 hide_index=True)
else:
    st.warning("No staff found in this video.")

timeline = alt.Chart(per_frame.assign(present=per_frame.staff_present.astype(int))).mark_area(
    color="#2e9e44", opacity=0.5).encode(x=alt.X("frame:Q", title="Frame"), y=alt.Y("present:Q", axis=None))
st.altair_chart(timeline.properties(height=80), width="stretch")

st.subheader("Task 2: staff xy coordinates")
left, right = st.columns(2)
with left:
    st.dataframe(present[["frame", "x", "y", "tag_visible", "tag_x", "tag_y"]].round(1), hide_index=True, height=360)
with right:
    path = alt.Chart(present).mark_circle(size=18).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[0, info["width"]])),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[info["height"], 0])),
        color=alt.Color("frame:Q", scale=alt.Scale(scheme="viridis")), tooltip=["frame", "x", "y"])
    st.altair_chart(path.properties(height=360, title="Staff path (image pixels)"), width="stretch")

st.subheader("Annotated video")
out = CACHE / f"{key}.mp4"
if not out.exists():
    vbar = st.progress(0.0, text="Rendering…")
    render_video(video, per_frame, out, progress=show_progress(vbar, "Rendering frame"))
    vbar.empty()
_, mid, _ = st.columns([1, 2, 1])
mid.video(str(out))
