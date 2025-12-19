import cv2
import numpy as np

class BallTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.initialized = False
        self.missed_frames = 0
        self.max_missed_visible = 10

        # ✅ ADD THESE
        self.init_count = 0
        self.init_threshold = 3

    def update(self, detection):
        """`
        Parameters:
        - detection: (x, y) or None

        Returns:
        - x, y, visible
        """

        # -------- Initialization phase (NO predict here!) --------
        if not self.initialized:
            if detection is not None:
                self.init_count += 1

                if self.init_count >= self.init_threshold:
                    # Initialize state
                    self.kf.statePost = np.array([
                        [np.float32(detection[0])],
                        [np.float32(detection[1])],
                        [0.0],
                        [0.0]
                    ], dtype=np.float32)

                    self.initialized = True
                    self.missed_frames = 0

                    return detection[0], detection[1], 1

            # Not initialized yet → invisible
            return -1, -1, 0

        # -------- Normal tracking phase --------
        # 1. Predict
        prediction = self.kf.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])

        # 2. Gating (only after initialization)
        if detection is not None:
            dx = detection[0] - pred_x
            dy = detection[1] - pred_y

            if dx * dx + dy * dy > 140 * 140:
                detection = None

        # 3. Correct if detection is valid
        if detection is not None:
            meas = np.array([
                [np.float32(detection[0])],
                [np.float32(detection[1])]
            ], dtype=np.float32)

            self.kf.correct(meas)
            self.missed_frames = 0
            return detection[0], detection[1], 1

        # 4. Prediction-only (short gaps)
        self.missed_frames += 1
        if self.missed_frames <= self.max_missed_visible:
            return pred_x, pred_y, 1

        # 5. Ball lost
        return -1, -1, 0