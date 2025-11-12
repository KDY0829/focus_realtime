# src/run_realtime_web.py
import os, time, cv2, threading
import numpy as np
from collections import deque
from datetime import datetime

from gaze import GazeEstimator
from neck import HeadPoseEstimator
from blink import BlinkEstimator
from emotion import EmotionPredictor
from fusion import concentration_score

from logger import SafeCSVLogger, StateCheckpoint
from web_server import create_app, run_server

SAVE_DIR = "runs"
SESSION = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(SAVE_DIR, SESSION)
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH  = os.path.join(OUT_DIR, "focus_log.csv")
CKPT_PATH = os.path.join(OUT_DIR, "state.json")

DURATION_SEC = 600      # 10분
FPS_TARGET   = 30

# 웹 대시보드 공유 상태
shared = {
    "lock": threading.Lock(),
    "time":  deque(maxlen=600*FPS_TARGET),
    "gaze":  deque(maxlen=600*FPS_TARGET),
    "neck":  deque(maxlen=600*FPS_TARGET),
    "emotion": deque(maxlen=600*FPS_TARGET),
    "blink": deque(maxlen=600*FPS_TARGET),
    "focus": deque(maxlen=600*FPS_TARGET),
    "latest": {},
    "fps": 0.0,
    "start_ts": SESSION,
    "frames": 0,
    "saved_rows": 0,
}

def overlay(frame, scores, final):
    h, w = frame.shape[:2]
    y = 28
    for k,v in scores.items():
        cv2.putText(frame, f"{k}: {v:.2f}", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        y += 26
    bar_w = int(final * (w - 20))
    cv2.rectangle(frame, (10, h-40), (10+bar_w, h-10), (0,200,255), -1)
    cv2.putText(frame, f"Focus: {final:.2f}", (10, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
    return frame

def main():
    # 모델들 초기화
    gaze = GazeEstimator()
    neck = HeadPoseEstimator()
    blink = BlinkEstimator()
    emo  = EmotionPredictor("models/emotion_classifier.pth")

    # 로거/체크포인트
    logger = SafeCSVLogger(
        CSV_PATH,
        header=["time_s","gaze","neck","emotion","blink","ear","blink_rate","yaw","pitch","roll","final"]
    )
    ckpt = StateCheckpoint(CKPT_PATH)
    ckpt.load()

    # 웹 서버 시작 (백그라운드)
    app = create_app(shared_state=shared, static_dir="../static")
    th = threading.Thread(target=run_server, args=(app,"127.0.0.1",8000), daemon=True)
    th.start()
    print("🌐 Dashboard: http://127.0.0.1:8000")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라 열기 실패")
        return

    t0 = time.time()
    fps_q = deque(maxlen=30)
    last_ckpt_save = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            t = time.time() - t0
            if t > DURATION_SEC:
                break

            g_score, g_off      = gaze.score(frame)
            n_score, angles     = neck.score(frame)
            b_score, ear, brate = blink.score(frame)
            e_score, e_probs, et = emo.predict(frame)

            final = concentration_score(g_score, n_score, e_score, b_score)

            # 오버레이
            scores = {
                "gaze": g_score or 0.0,
                "neck": n_score or 0.0,
                "emotion": e_score or 0.0,
                "blink": b_score or 0.0
            }
            disp = overlay(frame.copy(), scores, final)
            cv2.imshow("Focus Realtime", disp)

            # CSV 즉시 저장
            yaw, pitch, roll = (angles or (0,0,0))
            logger.write([
                round(t,2),
                scores["gaze"], scores["neck"], scores["emotion"], scores["blink"],
                round(ear or 0.0, 4), round(brate or 0.0, 2),
                round(yaw,2), round(pitch,2), round(roll,2),
                final
            ])
            shared["saved_rows"] += 1

            # 공유 상태 업데이트(웹 대시보드)
            with shared["lock"]:
                shared["time"].append(t)
                shared["gaze"].append(scores["gaze"])
                shared["neck"].append(scores["neck"])
                shared["emotion"].append(scores["emotion"])
                shared["blink"].append(scores["blink"])
                shared["focus"].append(final)
                shared["latest"] = {
                    "off_dx_dy": g_off.tolist() if g_off is not None else None,
                    "angles": {"yaw":yaw, "pitch":pitch, "roll":roll},
                    "ear": ear, "blink_rate": brate, "emotion_top": et
                }
                shared["frames"] += 1

            # 체크포인트(5초마다)
            if time.time() - last_ckpt_save > 5:
                ckpt.save({"frames": shared["frames"], "blinks": blink.blinks})
                last_ckpt_save = time.time()

            # FPS 조절 & 측정
            fps_q.append(time.time())
            if len(fps_q) >= 2:
                fps = len(fps_q) / (fps_q[-1]-fps_q[0])
                shared["fps"] = fps
                if fps > FPS_TARGET:
                    time.sleep(max(0, 1.0/FPS_TARGET))

            # 종료키
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.close()
        ckpt.save({"frames": shared["frames"], "blinks": blink.blinks})
        print(f"✅ CSV 저장: {CSV_PATH}")
        print(f"✅ 상태 체크포인트: {CKPT_PATH}")
        print("ℹ️ 웹 대시보드는 창을 닫으면 종료됩니다.")

if __name__ == "__main__":
    main()

