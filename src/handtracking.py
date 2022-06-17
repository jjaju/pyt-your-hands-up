import cv2
import mediapipe as mp
import time
import numpy as np
from typing import Tuple, List


class HandDetector():
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) -> None:
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_hands = mp.solutions.mediapipe.python.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.static_image_mode,
            self.max_num_hands,
            self.model_complexity,
            self.min_detection_confidence,
            self.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.mediapipe.python.solutions.drawing_utils

        # mediapipe landmark indices for finger tips
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img: np.ndarray, draw=True) -> np.ndarray:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        self.right_left = []

        if(self.results.multi_hand_landmarks):
            hands_temp = []
            for hand in self.results.multi_handedness:
                hands_temp.append(hand.classification[0].label)

            self.right_left.append("Right" in hands_temp)
            self.right_left.append("Left" in hands_temp)

            if draw:
                for hand_lms in self.results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_lms,
                        self.mp_hands.HAND_CONNECTIONS
                    )
        else:
            self.right_left = [False, False]

        return img

    def find_landmark_positions(self, img: np.ndarray) -> Tuple[List, List]:
        h, w, _ = img.shape
        self.lm_list_right = []
        self.lm_list_left = []

        if self.results.multi_handedness:
            for hand_index, hand in enumerate(self.results.multi_handedness):
                if hand.classification[0].label == "Right":
                    temp_list = self.lm_list_right
                else:
                    temp_list = self.lm_list_left

                lms = self.results.multi_hand_landmarks[hand_index].landmark
                for id, lm in enumerate(lms):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    temp_list.append([id, cx, cy])

        return self.lm_list_right, self.lm_list_left

    def fingers_extended(self, hand: str) -> List:
        fingers = []

        # Thumb: horizontal test, depends on type of hand (left or right)
        if hand == "Right":
            search_list = self.lm_list_right
            if (
                search_list[self.tip_ids[0]][1] <
                search_list[self.tip_ids[0] - 2][1]
            ):
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            search_list = self.lm_list_left
            if (
                search_list[self.tip_ds[0]][1] >
                search_list[self.tip_ids[0] - 2][1]
            ):
                fingers.append(1)
            else:
                fingers.append(0)

        # Other fingers: vertical test
        for id in range(1, 5):
            if (
                search_list[self.tip_ids[id]][2] <
                search_list[self.tip_ids[id] - 2][2]
            ):
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


def main():
    previous_time = 0

    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        img = detector.find_hands(img)
        lm_list_right, lm_list_left = detector.find_landmark_positions(img)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(
            img,
            str(int(fps)), (10, 20),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            1
        )
        cv2.imshow("Hand Detection", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
