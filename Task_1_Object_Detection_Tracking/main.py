
"""
main.py - Task 4: Object Detection and Tracking
 
Pipeline:
  1. Real-time video input (webcam or video file) using OpenCV
  2. Object detection with a pre-trained YOLO model (Ultralytics YOLOv8)
  3. Every frame is processed: detect objects -> bounding boxes
  4. Object tracking with SORT (see sort.py)
  5. Output is displayed live with class labels and tracking IDs
 
Examples:
  python main.py                       # default webcam
  python main.py --source 1            # second webcam
  python main.py --source video.mp4    # video file
  python main.py --source video.mp4 --save output.mp4
  python main.py --classes 0 2         # only person (0) and car (2)
 
Press  q  or  ESC  in the video window to quit.
"""
 
import argparse
import os
import sys
import time
 
import cv2
import numpy as np
 
from sort import Sort
 
 
def parse_args():
    p = argparse.ArgumentParser(description="YOLO object detection + SORT tracking")
    p.add_argument("--source", default="0",
                   help="Webcam index (0, 1, ...) or path to a video file. Default: 0")
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLO weights (auto-downloaded on first run). "
                        "Options: yolov8n.pt (fastest), yolov8s.pt, yolov8m.pt, ...")
    p.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold for the detector")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", default=None,
                   help="Device: 'cpu', '0' (first GPU) ... Default: auto")
    p.add_argument("--classes", type=int, nargs="+", default=None,
                   help="Filter by class IDs, e.g. --classes 0 2 (COCO: 0=person, 2=car)")
    p.add_argument("--max-age", type=int, default=20,
                   help="SORT: frames to keep a lost track alive")
    p.add_argument("--min-hits", type=int, default=3,
                   help="SORT: detections needed before a track is shown")
    p.add_argument("--track-iou", type=float, default=0.3,
                   help="SORT: IoU threshold for matching detections to tracks")
    p.add_argument("--save", default=None, help="Optional output video path, e.g. output.mp4")
    p.add_argument("--no-display", action="store_true",
                   help="Do not open a window (useful with --save)")
    return p.parse_args()
 
 
def open_capture(source: str):
    """Open a webcam (index) or a video file."""
    if source.isdigit():
        idx = int(source)
        cap = None
        # On Windows, the DirectShow backend is usually the most reliable for webcams
        if os.name == "nt":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = None
        if cap is None:
            cap = cv2.VideoCapture(idx)
        return cap
 
    if not os.path.isfile(source):
        print(f"[ERROR] Video file not found: {source}")
        return None
    return cv2.VideoCapture(source)
 
 
def color_for_id(track_id: int):
    """Return a stable, bright BGR color for a track ID."""
    rng = np.random.RandomState(int(track_id) * 9973 % (2 ** 32 - 1))
    return tuple(int(c) for c in rng.randint(60, 255, size=3))
 
 
def draw_track(frame, box, track_id, label, conf):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 0, w - 1))
    y1, y2 = int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 0, h - 1))
    color = color_for_id(track_id)
 
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
 
    text = f"ID {int(track_id)} | {label} {conf:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    top = max(y1 - th - baseline - 4, 0)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + baseline + 4), color, -1)
    cv2.putText(frame, text, (x1 + 3, top + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
 
 
def main():
    args = parse_args()
 
    # Import here so a missing package gives a clear message
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 'ultralytics' is not installed.\n"
              "        Run:  pip install -r requirements.txt")
        sys.exit(1)
 
    print(f"[INFO] Loading model: {args.model} (downloads automatically on first run)")
    model = YOLO(args.model)
    names = model.names  # {class_id: class_name}
 
    cap = open_capture(args.source)
    if cap is None or not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.source}")
        print("        - Webcam: close other apps using the camera, or try --source 1")
        print("        - File: check the path is correct")
        sys.exit(1)
 
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps != src_fps or src_fps <= 0 or src_fps > 240:
        src_fps = 30.0
 
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, src_fps, (width, height))
        if not writer.isOpened():
            print(f"[WARNING] Could not create output file: {args.save}")
            writer = None
 
    tracker = Sort(max_age=args.max_age, min_hits=args.min_hits,
                   iou_threshold=args.track_iou)
 
    window = "Object Detection and Tracking (press q to quit)"
    if not args.no_display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
 
    print("[INFO] Running... press 'q' or ESC in the video window to quit.")
    fps_smooth = 0.0
    frame_idx = 0
    is_file = not args.source.isdigit()
 
    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                if is_file:
                    print("[INFO] End of video.")
                else:
                    print("[WARNING] Failed to read frame from webcam.")
                break
            frame_idx += 1
 
            # ---- 1) Detection (YOLO) ----
            results = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                classes=args.classes,
                device=args.device,
                verbose=False,
            )[0]
 
            dets = np.empty((0, 6), dtype=np.float64)
            if results.boxes is not None and len(results.boxes) > 0:
                xyxy = results.boxes.xyxy.cpu().numpy()
                conf = results.boxes.conf.cpu().numpy().reshape(-1, 1)
                cls = results.boxes.cls.cpu().numpy().reshape(-1, 1)
                dets = np.hstack([xyxy, conf, cls]).astype(np.float64)
 
            # ---- 2) Tracking (SORT) ----
            tracks = tracker.update(dets)
 
            # ---- 3) Draw output ----
            for x1, y1, x2, y2, tid, cid, cf in tracks:
                label = names.get(int(cid), str(int(cid)))
                draw_track(frame, (x1, y1, x2, y2), tid, label, cf)
 
            # FPS (smoothed)
            dt = max(time.time() - t0, 1e-6)
            fps_smooth = (1.0 / dt) if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * (1.0 / dt)
            info = f"FPS: {fps_smooth:.1f} | Tracked objects: {len(tracks)}"
            cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
 
            if writer is not None:
                writer.write(frame)
 
            if not args.no_display:
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[INFO] Quit requested.")
                    break
                # Stop if the user closed the window with the X button
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
 
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[INFO] Saved output video to: {args.save}")
        cv2.destroyAllWindows()
        print(f"[INFO] Processed {frame_idx} frames. Done.")
 
 
if __name__ == "__main__":
    main()
 
