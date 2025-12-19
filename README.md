# Cricket Ball Detection and Tracking

This repository implements a complete computer vision pipeline to detect and track a cricket ball in videos captured from a single static camera, as part of the EdgeFleet.ai AI/ML assessment.

The system outputs:
- Per-frame ball centroid annotations (CSV)
- A processed video with ball centroid and trajectory overlay
- Fully reproducible inference code with no training required

## Problem Overview

Tracking a cricket ball is challenging due to:
- Small object size
- High speed and motion blur
- Frequent occlusions
- Visual similarity to bats and player highlights

Given the constraint that the provided dataset must not be used for training, this solution relies on a classical computer vision pipeline combined with Kalman filter–based tracking.

## Approach Summary

The pipeline consists of:

1. Motion Detection  
   Uses frame differencing / background subtraction (static camera assumption)

2. Color Filtering  
   HSV-based filtering to isolate white cricket balls

3. Shape & Size Constraints  
   Area and circularity filtering to reject players and bats

4. Kalman Filter Tracking  
   Constant-velocity motion model with delayed initialization

5. Visibility Logic  
   Handles short occlusions and missed detections

6. Visualization  
   Green dot: ball centroid  
   Blue polyline: ball trajectory

## Repository Structure

edgefleet-cricket-ball-tracking/  
│  
├── code/  
│   ├── infer.py        # Main inference pipeline  
│   ├── detect.py       # Ball detection logic  
│   └── track.py        # Kalman filter tracking  
│  
├── annotations/  
│   └── 1.csv     # Per-frame annotations
│  
├── results/  
│   └── 1.mp4     # Output video  
│  
├── requirements.txt  
├── README.md  
└── report.pdf  

## Dataset

The dataset provided by EdgeFleet.ai is used only for testing, as per the assignment instructions.

Download link:  
https://drive.google.com/file/d/1hnaGuqGuMXaFKI5fhfy8gatzCH-6iMcJ/

After downloading, place videos locally, for example:

data/videos/1.mp4

## Setup Instructions

1. Create a virtual environment (recommended):

python3 -m venv venv  
source venv/bin/activate  

2. Install dependencies:

pip install -r requirements.txt  

## Running Inference

python code/infer.py --video data/videos/1.mp4 --out_video results/1_output.mp4 --out_csv annotations/1.csv

## Output Format

CSV annotation file (annotations/*.csv):

frame,x,y,visible  
0,512,298,1  
1,518,305,1  
2,-1,-1,0  

- x, y = -1 when the ball is not visible
- visible = 1 if detected or confidently tracked

Processed video (results/*.mp4):
- Green dot: ball centroid
- Blue line: recent ball trajectory

## Debugging Notes

During development:
- Foreground masks were visualized to analyze false positives
- Detection sparsity vs. noise trade-offs were tuned iteratively
- Tracker initialization was delayed to avoid false starts
- Distance-based gating rejected implausible detections

Details are documented in report.pdf.

## Limitations

- Performance may degrade under extreme motion blur
- White bats close to the ball can still cause ambiguity
- HSV thresholds may require tuning for different lighting conditions
- No explicit bounce or physics-based modeling is used

## Conclusion

This project demonstrates that a carefully engineered classical computer vision pipeline, combined with temporal filtering, can robustly track a fast-moving cricket ball without any training data. The system is modular, reproducible, and suitable for real-time or edge deployment scenarios.

## Author

Kunal Kamalkishor Bhosikar  
M.S. (by Research), Computer Science  
IIIT Hyderabad
