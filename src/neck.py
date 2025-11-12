# src/neck.py
import cv2, numpy as np, mediapipe as mp

class HeadPoseEstimator:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)

        # 3D 참조점 (단위: 임의 스케일) - 고전 6점 세트
        self.model_3d = np.array([
            [0.0,   0.0,   0.0],     # nose tip (index 1)
            [0.0,  -63.6, -12.5],    # chin (152)
            [-43.3, 32.7, -26.0],    # left eye corner (33)
            [ 43.3, 32.7, -26.0],    # right eye corner (263)
            [-28.9,-28.9, -24.1],    # left mouth corner (61)
            [ 28.9,-28.9, -24.1],    # right mouth corner (291)
        ], dtype=np.float64)

        self.idxs = [1, 152, 33, 263, 61, 291]

    def _euler_from_R(self, R):
        # OpenCV의 Rodrigues 결과로부터 yaw/pitch/roll 추출 (deg)
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        if not singular:
            x = np.degrees(np.arctan2(R[2,1], R[2,2]))  # pitch
            y = np.degrees(np.arctan2(-R[2,0], sy))     # yaw
            z = np.degrees(np.arctan2(R[1,0], R[0,0]))  # roll
        else:
            x = np.degrees(np.arctan2(-R[1,2], R[1,1]))
            y = np.degrees(np.aratan2(-R[2,0], sy))
            z = 0
        return x, y, z

    def score(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        res = self.face_mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None, None

        lm = res.multi_face_landmarks[0].landmark
        pts2d = np.array([[lm[i].x * w, lm[i].y * h] for i in self.idxs], dtype=np.float64)

        # 카메라 행렬 (근사): f ~ max(w,h), cx=w/2, cy=h/2
        f = max(h, w)
        cam_mtx = np.array([[f, 0, w/2],
                            [0, f, h/2],
                            [0, 0,   1 ]], dtype=np.float64)
        dist = np.zeros((4,1))

        ok, rvec, tvec = cv2.solvePnP(self.model_3d, pts2d, cam_mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: return None, None
        R, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = self._euler_from_R(R)

        # 정상 범위(튜닝 가능): |yaw|<=25°, |pitch|<=20°, |roll|<=20°
        s = 1.0 - (abs(yaw)/25 + abs(pitch)/20 + abs(roll)/20)/3
        score = float(np.clip(s, 0.0, 1.0))
        return score, (yaw, pitch, roll)

