"""
Run with:  python run.py

NOTE: reload=False on purpose. uvicorn's --reload spawns a watcher process
that can restart the worker process, which would try to open the webcam a
second time and conflict with the still-running FaceEngine thread.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
