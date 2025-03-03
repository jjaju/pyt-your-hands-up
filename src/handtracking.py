import time
from pathlib import Path
from typing import List

import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmark,
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
)


class HandDetector:
    def __init__(
        self,
        model_path: str,
        running_mode=VisionTaskRunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) -> None:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=Path(model_path)),
            running_mode=running_mode,
            result_callback=self.return_result,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.hand_landmark = HandLandmark
        self.latest_detection = None

    def return_result(
        self,
        result: HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        self.latest_detection = result

    def detect(self, img: np.ndarray) -> HandLandmarkerResult | None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        timestamp = int(round(time.time() * 1000))
        self.landmarker.detect_async(mp_image, timestamp)
        return self.latest_detection

    def get_landmark_positions(
        self, detection_result: HandLandmarkerResult
    ) -> List:
        landmarks = []
        if len(detection_result.hand_landmarks) <= 0:
            return landmarks

        for l, landmark in enumerate(detection_result.hand_landmarks[0]):
            landmarks.append([l, landmark.x, landmark.y])

        return landmarks

    def transform_landmark_positions_to_image(
        self, landmarks: List, img: np.ndarray
    ) -> List:
        h, w, _ = img.shape
        transformed_landmarks = [
            [l, int(x * w), int(y * h)] for l, x, y in landmarks
        ]
        return transformed_landmarks

    def draw_landmarks(
        self, img: np.ndarray, detection_result: HandLandmarkerResult
    ) -> np.ndarray:
        hand_landmarks_list = (
            detection_result.hand_landmarks if detection_result else []
        )
        annotated_image = np.copy(img)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in hand_landmarks
                ]
            )
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style(),
            )

        return annotated_image
