# FootfallCam Assessment: Staff Identification

Finds the staff member (the person wearing a striped name tag) in a top-down
FootfallCam video.

- **Task 1:** which frames the staff member appears in
- **Task 2:** their xy position in each of those frames

## Approach

```
video ─► find people ─► track people ─► find the name tag ─► pick the staff member ─► frames, xy + annotated video
```

1. **Find people.** A YOLO26n person detector. The camera looks straight
   down through a fisheye lens, so people appear at every angle. Each frame
   is also checked rotated by 90°, 180° and 270°, and the results are
   combined.
2. **Track people.** The BoT-SORT tracker gives each person an ID that stays
   the same from frame to frame.
3. **Find the name tag.** A second YOLO26n, fine-tuned to detect the tag. The
   tag is tiny (about 12×5 px), so this model was trained on 55 hand-checked
   tag positions from one sample video. Each tag is assigned to the one
   person it belongs to.
4. **Pick the staff member.** A person becomes staff once the tag is seen on
   them at least twice, with at least one clear sighting. The label then
   stays with that person for up to 100 frames (4 s) either side of a
   sighting, so they are still followed when they turn and the tag faces
   away. Boxes that would need an impossible jump between frames are dropped,
   keeping the path with the most tag sightings. Gaps of up to 50 frames (2 s)
   are filled in, using boxes of moving people where they fit and estimated
   positions elsewhere.

The xy position is the centre of the staff member's bounding box, in image
pixels.

## Project structure

```
app.py               Web dashboard (upload a video, see the results)
staffid/
  detect.py          Person detection
  track.py           Person tracking
  tag.py             Name-tag detection
  pipeline.py        Runs the steps above and draws the annotated video
models/              yolo26n.pt (pretrained), tag_yolo26n.pt (trained tag detector)
```

## How to run

Requires Python 3 and ffmpeg (for browser-playable video).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Then start the dashboard and upload a video:

```bash
.venv/bin/streamlit run app.py
```

The first run takes about 3 minutes for a 54-second video on an Apple Silicon
Mac, and longer without a GPU. The results are saved, so reopening it is instant
until you switch to another video or change the code.

## Output

The dashboard shows, for the uploaded video:

- **Task 1:** the frame ranges with the staff member, and a timeline
- **Task 2:** a table of their xy position per frame, and a plot of their path
- **Annotated video:** the staff member boxed in green and the tag circled in
  yellow

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
