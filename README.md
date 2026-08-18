# Distraction Monitor

A desktop app that watches your webcam during a work session and nags you — video and audio, overlaid directly on
the feed — whenever you're distracted, until you refocus.

"Distracted" means either of two things: your face is turned or tilted away from the screen past a calibrated
angle, or a phone is visible in frame. Either one alone triggers the alert. When you're back on task, the alert
clears on its own.

## How it works

- **Face-away detection** — [MediaPipe `FaceLandmarker`](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
  (synchronous `IMAGE` mode) extracts six facial landmarks (nose tip, chin, eye corners, mouth corners), which feed
  a `cv2.solvePnP` head-pose estimate. At the start of each session the app calibrates a neutral pitch/yaw baseline
  while you look at the screen normally, then flags "away" whenever live pitch/yaw drifts past a threshold relative
  to that baseline.
- **Phone detection** — a YOLOv8n model ([`ultralytics`](https://github.com/ultralytics/ultralytics)) runs directly
  on the full webcam frame. Ships with stock COCO-pretrained weights (already knows what a phone looks like) and can
  be fine-tuned on your own labeled dataset for better accuracy in your specific setup.
- **Debounced state machine** — a handful of consecutive distracted frames triggers the alert (phone reacts faster
  than face-away, since head pose is the noisier signal of the two); a longer run of clear frames is required to
  dismiss it. This avoids the alert flickering on/off from single-frame misdetections.
- **The alert** — a short video clip loops as a picture-in-picture overlay directly on the webcam feed (darkened
  background, bordered inset, banner text), with its audio track looped in sync. `OpenCV` has no audio support at
  all, so video playback and audio playback are two independent pipelines kept in sync by the app, not one.
- **The UI** — a single-window Tkinter app. Camera access only ever opens when you click **Start Session** (that
  click is what triggers the OS permission prompt). All capture and inference run on a background thread; the UI
  polls the latest processed frame on a timer. **Stop Session** ends with a summary: total session time, focused
  time, distracted time, and number of distraction events.

## Project layout

```
distraction-monitor/
├── src/
│   ├── app.py                   # entry point - the Tkinter desktop app
│   ├── monitor.py               # CLI variant - same detection loop, plain OpenCV windows, no UI chrome
│   ├── face_tracker.py          # MediaPipe FaceLandmarker wrapper
│   ├── head_pose_estimator.py   # solvePnP-based pitch/yaw + calibration
│   ├── phone_detector.py        # YOLOv8 wrapper
│   ├── gif_player.py            # loops the alert clip's video frames
│   ├── audio_player.py          # extracts + loops the alert clip's audio
│   ├── convert_csv_to_yolo.py   # dataset prep for fine-tuning phone_detector's model
│   └── train_phone_yolo.py      # fine-tuning script (Colab-friendly)
├── model/                       # face landmark model + YOLO weights
├── assets/                      # the alert video clip
├── pyproject.toml / uv.lock
└── LICENSE
```

## Getting started

Requires Python 3.14 and a webcam.

```bash
uv sync
uv run src/app.py
```

(Or `pip install -e .` / `python src/app.py` if you're not using `uv`.)

Click **Start Session**, look at your screen for a few seconds while it calibrates, then work normally. Click
**Stop Session** when you're done to see the summary.

`src/monitor.py` runs the same detection logic without the UI — useful for debugging the face/phone signals in
isolation, with live bounding boxes and status text in a plain OpenCV window.

## Fine-tuning the phone detector

The bundled YOLOv8n weights are pretrained on COCO and already recognize phones reasonably well out of the box. To
improve accuracy for your own camera/lighting/angle:

1. Label a dataset (CSV with `filename,width,height,xmin,ymin,xmax,ymax,class` columns, uncropped images).
2. Run `python src/convert_csv_to_yolo.py` to convert it into YOLO training format.
3. Run `python model/train/distraction_monitor.ipynb` to fine-tune (works well on a free Colab GPU runtime).
4. Point `phone_detector.py`'s `MODEL_PATH` at the resulting weights.

## Design notes worth knowing

A few non-obvious decisions shaped this codebase, in case you're reading the source or extending it:

- **Synchronous MediaPipe, not async/`LIVE_STREAM`.** Reading "the latest callback result" immediately after
  submitting a new frame is a race condition — it can silently return a stale result from several frames back. Both
  the face and (earlier, now-removed) hand landmark pipelines run in synchronous `IMAGE` mode instead, so every call
  is guaranteed to return landmarks for the exact frame just submitted.
- **No `solvePnP` warm-starting.** Pose estimation solves fresh every frame rather than seeding from the previous
  frame's rotation/translation vectors. With only six correspondence points and an uncalibrated camera matrix, a
  single noisy frame can land in a bad local minimum — warm-starting from it means every later frame refines from
  the same wrong pose and never recovers.
- **Full-frame detection, not crop-then-classify.** An earlier version of this project cropped a hand region and
  classified the crop as phone/pen/none. It was abandoned: any mismatch in how crops are generated across different
  data sources (or between training data and live inference) teaches the model to distinguish *which pipeline
  produced an image* rather than the object itself. Running a detector on the full, uncropped frame — matching
  training data to inference input exactly — sidesteps that entire failure class.
- **Video and audio are separate pipelines.** `cv2.VideoCapture`/`imshow` has zero audio support. The alert clip's
  audio is extracted once via `ffmpeg` and looped independently, kept in sync with the video loop through the same
  state-machine transitions rather than any shared timer.

## License

MIT — see [LICENSE](LICENSE).
