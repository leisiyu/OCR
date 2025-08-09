import cv2
import pytesseract
import json
import os


def extract_keyframes_and_ocr(video_path, frame_interval=30, output_json="ocr_output.json", roi=None):
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
            if roi:
                x, y, w, h = roi
                frame = frame[y:y+h, x:x+w]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            text = pytesseract.image_to_string(rgb_frame)

            results.append({
                "timestamp": round(timestamp_sec, 2),
                "text": text.strip()
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
        roi=(1000, 0, 400, 150)  # <-- change this to your area
    )
