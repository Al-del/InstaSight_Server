import eventlet
eventlet.monkey_patch()

import asyncio
import time
import cv2
import numpy as np
from flask import Flask, Response
from flask_socketio import SocketIO
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from threading import Lock
from utilss import generate_bbox, process_image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s").to(device)
feature_extractor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf").to(device)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(
    app,
    async_mode='eventlet',
    cors_allowed_origins=["http://localhost:4200"]
)

pcs = set()
latest_frame = None
frame_lock = Lock()

# Create a dedicated asyncio event loop
asyncio_loop = asyncio.new_event_loop()

def run_async(coro):
    """Run an asyncio coroutine from eventlet"""
    future = asyncio.run_coroutine_threadsafe(coro, asyncio_loop)
    return future.result()

@app.route('/')
def index():
    return "WebRTC signaling server is running."

@app.route('/videos')
def video_feed():
    """Route that serves the latest video frame as MJPG stream"""
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                print("No frame yet. Sending black frame.")
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', blank_frame)
                frame = buffer.tobytes()
            else:
                print("Sending processed frame.")
                _, buffer = cv2.imencode('.jpg', latest_frame)
                frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)


@socketio.on('connect')
def handle_connect():
    print("\n=== New client connected ===")

@socketio.on('disconnect')
def handle_disconnect():
    print("\n=== Client disconnected ===")

@socketio.on('message')
def handle_message(data):
    print("\nIncoming message:", data)
    msg_type = data.get("type")
    
    if msg_type == "offer":
        print("Processing offer...")
        socketio.start_background_task(run_async, handle_offer(data))
    elif msg_type == "ice-candidate":
        print("Processing ICE candidate...")
        socketio.start_background_task(run_async, handle_ice_candidate(data))

async def handle_offer(data):
    print("\n=== Handling Offer ===")
    offer = data.get("offer")
    if not offer:
        print("Error: No offer in data")
        return

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        state = pc.iceConnectionState
        print(f"\nICE Connection State: {state}")
        if state == "failed":
            print("ICE Connection Failed!")
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        print(f"\nTrack received! Kind: {track.kind}, ID: {track.id}")
        
        if track.kind == "video":
            print("Starting video frame processing...")
            
            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                print(f"Connection state: {pc.connectionState}")
                if pc.connectionState == "connected":
                    print("WebRTC connection established!")
            
            async def process_frames():
                global latest_frame
                frame_count = 0
                last_log = time.time()
                try:
                    while True:
                        frame = await track.recv()
                        frame_count += 1
                        
                        if time.time() - last_log >= 1:
                            print(f"Receiving frames... {frame_count} frames received")
                            last_log = time.time()
                            
                        try:
                            img = frame.to_ndarray(format="bgr24")
                            if frame_count % 3 == 0:
                                # Get all return values including close_objects
                                processed_img, bbox, depth_map, close_objects = process_image(
                                    img, yolo_model, feature_extractor, device, depth_model
                                )
                                
                                # Send warning if any objects are too close
                                if close_objects:
                                    print("OK")
                                    socketio.emit('object_warning', {
                                        'warning': 'TOO_CLOSE',
                                        'objects': close_objects,
                                        'timestamp': time.time()
                                    })
                                
                                # Convert depth map for visualization if needed
                                depth_vis = (depth_map * 255).astype(np.uint8)
                                depth_vis = np.repeat(depth_vis[:, :, np.newaxis], 3, axis=2)
                                
                                # Store the processed image (not depth map) as latest_frame
                                latest_frame = processed_img
                                
                        except Exception as e:
                            print(f"Frame processing error: {e}")
                except Exception as e:
                    print(f"Frame processing error: {e}")
            asyncio.create_task(process_frames())

    try:
        print("\nSetting remote description...")
        await pc.setRemoteDescription(RTCSessionDescription(
            sdp=offer["sdp"],
            type=offer["type"]
        ))

        print("Creating answer...")
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        print("Sending answer back to client...")
        socketio.emit('message', {
            "type": "answer",
            "answer": {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            }
        })
        print("Answer sent successfully!")
        
    except Exception as e:
        print(f"\nError handling offer: {str(e)}")
        pcs.discard(pc)
        await pc.close()

async def handle_ice_candidate(data):
    print("\nHandling ICE candidate...")
    candidate_dict = data.get("candidate")
    if not candidate_dict:
        print("Error: No candidate in data")
        return

    if not pcs:
        print("Error: No peer connection available")
        return
        
    pc = next(iter(pcs))
    try:
        print(f"Adding ICE candidate: {candidate_dict}")
        await pc.addIceCandidate(RTCIceCandidate(
            candidate=candidate_dict["candidate"],
            sdpMid=candidate_dict["sdpMid"],
            sdpMLineIndex=candidate_dict["sdpMLineIndex"]
        ))
    except Exception as e:
        print(f"Error adding ICE candidate: {e}")

def run_asyncio_loop():
    asyncio.set_event_loop(asyncio_loop)
    asyncio_loop.run_forever()

if __name__ == '__main__':
    # Start the asyncio loop in a separate thread
    import threading
    threading.Thread(target=run_asyncio_loop, daemon=True).start()
    
    print("Starting server...")
    socketio.run(app, host='192.168.0.110', port=5000, debug=True)