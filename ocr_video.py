import cv2
import pytesseract
import json
import os


def extract_keyframes_and_ocr(video_path, frame_interval=30, output_json="ocr_output.json", roi=None, time_roi=None):
    """
    Extracts key frames from a video, performs OCR, and saves results to JSON.

    Args:
        video_path (str): Path to the video file.
        frame_interval (int): Process every Nth frame.
        output_json (str): Path to save JSON results.
        roi (tuple or None): (x, y, w, h) defining the region of interest for OCR.
                             If None, the whole frame is used.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video.")

    results = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp_sec = frame_count / fps

            # Apply ROI crop if provided
            text_frame = frame
            if roi:
                x, y, w, h = roi
                text_frame = frame[y:y+h, x:x+w]

            rgb_frame = cv2.cvtColor(text_frame, cv2.COLOR_BGR2RGB)
            text = pytesseract.image_to_string(rgb_frame)

            time_frame = frame
            if time_roi:
                x, y, w, h = time_roi
                time_frame = frame[y:y+h, x:x+w]
            rgb_time_frame = cv2.cvtColor(time_frame, cv2.COLOR_BGR2RGB)
            time = pytesseract.image_to_string(rgb_time_frame)

            results.append({
                "timestamp": round(timestamp_sec, 2),
                "text": text.strip(),
                "ingame_time": time.strip()
            })

        frame_count += 1

    cap.release()

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"OCR results saved to {output_json}")


# Example usage:
if __name__ == "__main__":
    # Extract OCR only from a box at (x=100, y=200) with width=400, height=150
    extract_keyframes_and_ocr(
        "sims4.mp4",
        frame_interval=240,
        roi=(2600, 0, 472, 400),  # change this to your area, default = None
        time_roi = (1400, 1800, 180, 100)  # change this to the in-game time area, default = None
    )
