import cv2
import numpy as np

class BallDetector:
    def __init__(
        self,
        min_area=30,
        max_area=180,
        min_circularity=0.65
    ):
        """
        Parameters:
        - min_area: minimum contour area to be considered ball
        - max_area: maximum contour area
        - min_circularity: shape filter (1 = perfect circle)
        """

        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity

        # Background subtractor (static camera assumption)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False
        )

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame):
        """
        Detect cricket ball in a single frame.

        Returns:
        - (x, y) centroid if detected
        - None if not detected
        """

        # 1. Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # 2. Morphological cleanup
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, self.kernel)

        # 3. Color filtering (WHITE ball)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 170])
        upper_white = np.array([180, 60, 255])

        color_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Combine motion + color
        fg_mask = cv2.bitwise_and(fg_mask, color_mask)

        cv2.imshow("fg_mask", fg_mask)
        cv2.waitKey(1)

        # 3. Find contours
        contours, _ = cv2.findContours(
            fg_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        best_candidate = None
        best_score = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Area filter
            if area < self.min_area or area > self.max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            # Circularity check
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            # Compute centroid
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Prefer smaller, more circular objects
            score = circularity - 0.002 * area

            if score > best_score:
                best_score = score
                best_candidate = (cx, cy)

        return best_candidate
