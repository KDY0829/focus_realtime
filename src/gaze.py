# src/gaze.py
import cv2, numpy as np, mediapipe as mp

class GazeEstimator:
    def __init__(self, iris_enabled=True):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=iris_enabled
        )
        # 눈/홍채 주요 인덱스 (MediaPipe FaceMesh)
        self.left_eye_corners  = [33, 133]
        self.right_eye_corners = [362, 263]
        # 홍채(왼/오) 468~471, 473~476
        self.left_iris  = [468, 469, 470, 471]
        self.right_iris = [473, 474, 475, 476]

    def _center(self, pts):
        pts = np.asarray(pts, dtype=np.float32)
        return pts.mean(axis=0)

    def _norm_offset(self, corners, iris):
        eye_center  = self._center(corners)
        iris_center = self._center(iris)
        # 눈 크기로 정규화
        eye_w = np.linalg.norm(corners[0] - corners[1]) + 1e-6
        off = (iris_center - eye_center) / eye_w  # (dx, dy)
        return off

    def score(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        res = self.face_mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None, None
        lm = res.multi_face_landmarks[0].landmark

        def xy(i): return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

        Lc = [xy(i) for i in self.left_eye_corners]
        Rc = [xy(i) for i in self.right_eye_corners]
        Li = [xy(i) for i in self.left_iris]
        Ri = [xy(i) for i in self.right_iris]

        offL = self._norm_offset(Lc, Li)
        offR = self._norm_offset(Rc, Ri)
        off  = (offL + offR) / 2.0  # 양안 평균

        # 중앙에서 멀수록 감점. 보정계수 k=2.0 (필요시 조절)
        dist = np.linalg.norm(off)
        gaze_score = float(np.clip(1.0 - dist * 2.0, 0.0, 1.0))
        return gaze_score, off  # score, (dx,dy)

