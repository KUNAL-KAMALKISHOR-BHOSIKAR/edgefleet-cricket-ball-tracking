import cv2
import csv
import argparse
from detect import BallDetector
from track import BallTracker


def run_inference(video_path, output_video_path, csv_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise IOError("Cannot open video writer")

    detector = BallDetector()
    tracker = BallTracker()

    frame_idx = 0
    trajectory = []

    with open(csv_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["frame", "x", "y", "visible"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detection = detector.detect(frame)
            x, y, visible = tracker.update(detection)

            print(frame_idx, detection is not None, visible)

            csv_writer.writerow([frame_idx, x, y, visible])

            if visible:
                trajectory.append((x, y))
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            if not visible:
                trajectory.clear()

            for i in range(1, len(trajectory)):
                cv2.line(
                    frame,
                    trajectory[i - 1],
                    trajectory[i],
                    (255, 0, 0),
                    2
                )

            writer.write(frame)
            frame_idx += 1

    cap.release()
    writer.release()
    print("Inference completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out_video", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    run_inference(args.video, args.out_video, args.out_csv)
