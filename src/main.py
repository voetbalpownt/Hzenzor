import cv2
import time
import json
import asyncio
import os
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "templates"), exist_ok=True)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

 
MODEL_PT = 'yolov8n.pt'
model = YOLO(MODEL_PT)

latest_detections = []
current_fps = 0.0

REFERENCE_HEIGHTS = {"person": 1.75, "boat": 2.5, "bottle": 0.25, "default": 1.0}
FOCAL_LENGTH_KM = 280

def estimate_distance(class_name, pixel_height):
    real_height = REFERENCE_HEIGHTS.get(class_name, REFERENCE_HEIGHTS["default"])
    if pixel_height < 5: return 0
    return round((real_height * FOCAL_LENGTH_KM) / pixel_height, 1)


def generate_frames():
    global latest_detections, current_fps
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    fps_timer = time.time()
    frame_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.01)
            continue
            
       
        results = model.predict(source=frame, imgsz=640, conf=0.45, verbose=False, device='cpu')
        
        this_frame_dets = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                dist = estimate_distance(label, (y2 - y1))
                
                x_center = (x1 + x2) / 2
                if x_center < 640 / 3:
                    positie = "Bakboord"
                elif x_center > 2 * 640 / 3:
                    positie = "Stuurboord"
                else:
                    positie = "Vooruit"
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {dist}m {positie}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                this_frame_dets.append({"label": label, "distance": dist, "conf": round(conf, 2), "position": positie})
                
      
        latest_detections = this_frame_dets
        
        frame_count += 1
        if time.time() - fps_timer >= 1.0:
            current_fps = frame_count / (time.time() - fps_timer)
            frame_count = 0
            fps_timer = time.time()
            
        _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + encoded.tobytes() + b'\r\n')
        
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    
@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
    
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json({
                "fps": round(current_fps, 1),
                "model": MODEL_PT,
                "detections": latest_detections
            })
            await asyncio.sleep(0.5)
    except: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
