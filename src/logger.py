# src/logger.py
import os, csv, json, time, threading
from datetime import datetime

class SafeCSVLogger:
    """끊겨도 손실 최소화를 위해 매 행 write + flush하는 CSV 로거"""
    def __init__(self, csv_path, header):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.csv_path = csv_path
        self._lock = threading.Lock()
        self._file = open(csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if os.stat(csv_path).st_size == 0:
            self._writer.writerow(header)
            self._file.flush()

    def write(self, row):
        with self._lock:
            self._writer.writerow(row)
            self._file.flush()  # 즉시 디스크 반영

    def close(self):
        try:
            self._file.close()
        except:
            pass

class StateCheckpoint:
    """런타임 상태(예: blink count, 세션 시간)를 주기적으로 저장/복구"""
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self.state = {"start_ts": None, "frames": 0, "blinks": 0}

    def load(self):
        if os.path.exists(self.ckpt_path):
            try:
                with open(self.ckpt_path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except:
                pass
        if self.state.get("start_ts") is None:
            self.state["start_ts"] = datetime.now().isoformat()

    def save(self, extra=None):
        if extra:
            self.state.update(extra)
        tmp = self.ckpt_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.ckpt_path)

