# src/blink.py
import numpy as np, cv2, mediapipe as mp
from collections import deque

class BlinkEstimator:
    def __init__(self, ear_close=0.18, ear_open=0.30, win=3):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
        self.left_idx  = [33, 160, 158, 133, 153, 144]   # (p1,p2,p3,p4,p5,p6)
        self.right_idx = [263, 387, 385, 362, 380, 373]
        self.ear_close = ear_close
        self.ear_open  = ear_open
        self.win = win
        self.hist = deque(maxlen=win)
        self.blinks = 0
        self.prev_closed = False

    def _ear(self, e):
        A = np.linalg.norm(e[1]-e[5])
        B = np.linalg.norm(e[2]-e[4])
        C = np.linalg.norm(e[0]-e[3]) + 1e-6
        return (A + B) / (2.0 * C)

    def score(self, frame_bgr, fps_hint=30.0):
        h, w = frame_bgr.shape[:2]
        res = self.face_mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None, None, None

        lm = res.multi_face_landmarks[0].landmark
        def xy(i): return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)
        L = np.array([xy(i) for i in self.left_idx])
        R = np.array([xy(i) for i in self.right_idx])
        ear = (self._ear(L) + self._ear(R)) / 2.0

        # 프레임 점수: EAR가 열림(open)에 가까울수록 1
        s = (ear - self.ear_close) / (self.ear_open - self.ear_close + 1e-6)
        frame_score = float(np.clip(s, 0.0, 1.0))

        # 깜빡임 카운트 (close→open)
        closed = ear < self.ear_close
        self.hist.append(closed)
        now_closed = sum(self.hist) >= (self.win//2 + 1)
        blink_evt = (self.prev_closed and not now_closed)
        if blink_evt:
            self.blinks += 1
        self.prev_closed = now_closed

        # 분당 깜빡임 수(대략값)
        blink_rate = self.blinks * (60.0 / max(1.0, len(self.hist)/fps_hint))
        return frame_score, ear, blink_rate

