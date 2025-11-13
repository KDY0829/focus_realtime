# src/web_server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

def create_app(shared_state, static_dir="../static"):
    app = FastAPI()

    # CORS 설정 (지금 Flask에서 하던 것과 동일한 역할)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # index.html 반환
    @app.get("/")
    def index():
        index_path = os.path.join(static_dir, "index.html")
        return FileResponse(index_path)

    # 최근 샘플들 반환
    @app.get("/api/live")
    def live():
        with shared_state["lock"]:
            data = {
                "time":       list(shared_state["time"])[-180:],
                "gaze":       list(shared_state["gaze"])[-180:],
                "neck":       list(shared_state["neck"])[-180:],
                "emotion":    list(shared_state["emotion"])[-180:],
                "blink":      list(shared_state["blink"])[-180:],
                "focus":      list(shared_state["focus"])[-180:],
                "fps":        shared_state.get("fps", 0.0),
                "latest":     shared_state.get("latest", {})
            }
        # FastAPI는 dict를 자동으로 JSON으로 반환해줌
        return data

    # 세션 메타 정보
    @app.get("/api/session")
    def session():
        with shared_state["lock"]:
            meta = {
                "start_ts": shared_state.get("start_ts"),
                "frames":   shared_state.get("frames", 0),
                "saved":    shared_state.get("saved_rows", 0)
            }
        return meta

    # (선택) /static 경로로 정적 파일 서빙 (지금 index.html은 상대경로만 쓰고 있어서 필수는 아님)
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    return app


def run_server(app, host="127.0.0.1", port=8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port, reload=False)
