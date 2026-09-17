import cv2
import numpy as np
import threading
import time
from datetime import datetime
from ultralytics import YOLO
import mediapipe as mp
from collections import deque
import json

class VideoProcessor:
    def __init__(self, video_path=None, camera_index=None, session_id=None, socketio=None, sport="soccer"):
        self.sport = str(sport or "soccer").lower()
        self.video_path = video_path
        self.camera_index = camera_index
        self.session_id = session_id
        self.socketio = socketio
        self.created_at = datetime.now()
        
        # Processing state
        self.is_processing = False
        self.is_live = camera_index is not None
        self.cap = None
        
        # Tracking data storage
        self.tracking_data = {
            'sport': self.sport,
            'players': [],
            'ball': [],
            'frames': [],
            'timestamps': []
        }
        
        # Initialize models
        self.initialize_models()
        
        # Performance metrics
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
    def initialize_models(self):
        """Initialize YOLO and MediaPipe models"""
        try:
            print("Loading YOLO model...", flush=True)

            self.yolo_model = YOLO("yolov8n.pt")

            print(
                "YOLO model loaded successfully: yolov8n.pt",
                flush=True
            )

            # MediaPipe Pose
            try:
                self.mp_pose = mp.solutions.pose

                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    smooth_segmentation=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )

                print(
                    "MediaPipe Pose initialized successfully",
                    flush=True
                )
            except Exception as pose_error:
                print(
                    f"MediaPipe initialization warning: {pose_error}",
                    flush=True
                )
                self.mp_pose = None
                self.pose = None

        except Exception as e:
            print(
                f"YOLO INITIALIZATION ERROR: {type(e).__name__}: {e}",
                flush=True
            )

            self.yolo_model = None
            self.mp_pose = None
            self.pose = None
    
    def start_live_processing(self, client_id):
        """Start live video processing"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            
            self.is_processing = True
            print(f"Started live processing for session {self.session_id}")
            
            while self.is_processing:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, analysis_data = self.process_frame(frame)
                
                # Send data to client
                if self.socketio:
                    self.socketio.emit('frame_data', {
                        'session_id': self.session_id,
                        'frame': self.frame_to_base64(processed_frame),
                        'analysis': analysis_data
                    }, room=client_id)
                
                # Control frame rate
                time.sleep(1/30)  # 30 FPS
                
        except Exception as e:
            print(f"Error in live processing: {e}")
            if self.socketio:
                self.socketio.emit('error', {'message': str(e)}, room=client_id)
        finally:
            self.stop()
    
    def start_upload_processing(self, client_id):
        """Start processing uploaded video"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise Exception(
                    "Could not open video file. Please upload a valid MP4, AVI, MOV, MKV, WEBM, M4V, MPEG, or MPG video."
                )
            
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.is_processing = True
            print(f"Started upload processing for session {self.session_id}")
            
            frame_count = 0
            while self.is_processing:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, analysis_data = self.process_frame(frame)
                
                # Update progress
                frame_count += 1
                progress = (frame_count / total_frames) * 100
                
                # Send data to client
                if self.socketio:
                    self.socketio.emit('frame_data', {
                        'session_id': self.session_id,
                        'frame': self.frame_to_base64(processed_frame),
                        'analysis': analysis_data,
                        'progress': progress,
                        'current_frame': frame_count,
                        'total_frames': total_frames
                    }, room=client_id)
                
                # Control processing speed
                time.sleep(1/fps)
                
        except Exception as e:
            print(f"Error in upload processing: {e}")
            if self.socketio:
                self.socketio.emit('error', {'message': str(e)}, room=client_id)
        finally:
            self.stop()
    
    def process_frame(self, frame):
        """Process a single frame and return analysis data"""
        try:
            # Store original frame dimensions
            height, width = frame.shape[:2]
            
            # Detect players and ball
            player_detections = self.detect_players(frame)
            ball_detection = self.detect_ball(frame)
            
            # Track objects
            tracked_players = self.track_players(player_detections, frame)
            tracked_ball = self.track_ball(ball_detection, frame)
            
            # Draw overlays
            processed_frame = self.draw_overlays(frame, tracked_players, tracked_ball)
            
            # Store tracking data
            frame_data = {
                'timestamp': time.time(),
                'players': tracked_players,
                'ball': tracked_ball,
                'frame_number': len(self.tracking_data['frames'])
            }
            
            self.tracking_data['frames'].append(frame_data)
            self.tracking_data['players'].append(tracked_players)
            self.tracking_data['ball'].append(tracked_ball)
            self.tracking_data['timestamps'].append(frame_data['timestamp'])
            
            # Calculate FPS
            self.update_fps()
            
            # Prepare analysis data
            analysis_data = {
                'fps': self.current_fps,
                'player_count': len(tracked_players),
                'ball_detected': ball_detection is not None,
                'frame_number': frame_data['frame_number'],
                'sport': self.sport
            }
            
            return processed_frame, analysis_data
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame, {}
    
        def detect_players(self, frame):
            """Detect people/players using YOLO person class."""
        try:
            if self.yolo_model is None:
                print("PLAYER DETECTION: YOLO model is not loaded", flush=True)
                return []

            height, width = frame.shape[:2]
            player_detections = []

            results = self.yolo_model(
                frame,
                verbose=False,
                conf=0.01,
                imgsz=1280
            )

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # COCO class 0 = person
                    if class_id != 0:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    x1 = max(0, min(int(x1), width - 1))
                    y1 = max(0, min(int(y1), height - 1))
                    x2 = max(0, min(int(x2), width - 1))
                    y2 = max(0, min(int(y2), height - 1))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    player_detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(confidence, 4),
                        "center": [
                            int((x1 + x2) / 2),
                            int((y1 + y2) / 2)
                        ]
                    })

            print(
                f"PLAYER DETECTION: {len(player_detections)} players detected",
                flush=True
            )
            print(f"RAW PLAYER COUNT: {len(player_detections)}", flush=True)

            return player_detections

        except Exception as e:
            print(f"PLAYER DETECTION ERROR: {e}", flush=True)
            return []
    def detect_ball(self, frame):
        """Detect a sports ball using YOLO sports-ball class plus sport-aware HSV fallback."""
        try:
            height, width = frame.shape[:2]
            candidates = []

            # COCO class 32 = sports ball. This is useful across sports but
            # remains experimental because the base YOLO model is generic.
            if self.yolo_model is not None:
                results = self.yolo_model(frame, verbose=False)
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    for box in boxes:
                        if int(box.cls[0]) == 32:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])
                            if conf >= 0.20:
                                candidates.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'center': [int((x1+x2)/2), int((y1+y2)/2)],
                                    'area': max(1, int((x2-x1)*(y2-y1))),
                                    'confidence': conf,
                                    'method': 'yolo_sports_ball'
                                })

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            ranges = {
                'soccer': [(np.array([0, 0, 170]), np.array([180, 55, 255]))],
                'basketball': [(np.array([3, 60, 50]), np.array([25, 255, 255]))],
                'tennis': [(np.array([25, 60, 80]), np.array([45, 255, 255]))],
                'cricket': [(np.array([0, 70, 50]), np.array([12, 255, 255])),
                            (np.array([165, 70, 50]), np.array([180, 255, 255]))],
                'volleyball': [(np.array([0, 0, 160]), np.array([180, 70, 255]))],
            }
            for lower, upper in ranges.get(self.sport, ranges['soccer']):
                mask = cv2.inRange(hsv, lower, upper)
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 30 or area > min(width * height * 0.02, 8000):
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    if w <= 0 or h <= 0:
                        continue
                    ratio = w / float(h)
                    if ratio < 0.45 or ratio > 2.2:
                        continue
                    perimeter = cv2.arcLength(contour, True)
                    circularity = (4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0
                    if circularity < 0.15:
                        continue
                    candidates.append({
                        'bbox': [x, y, x+w, y+h],
                        'center': [x+w//2, y+h//2],
                        'area': float(area),
                        'confidence': float(min(0.85, 0.35 + circularity * 0.5)),
                        'method': 'hsv'
                    })

            if not candidates:
                return None

            # Prefer YOLO when available; otherwise choose the most compact
            # plausible contour.
            yolo = [c for c in candidates if c.get('method') == 'yolo_sports_ball']
            best = max(yolo, key=lambda c: c['confidence']) if yolo else max(candidates, key=lambda c: c['confidence'])
            best.pop('method', None)
            return best

        except Exception as e:
            print(f"Error detecting {self.sport} ball: {e}")
            return None

    def detect_players(self, frame):
        """Detect players using YOLO."""
        try:
            if self.yolo_model is None:
                return []

            height, width = frame.shape[:2]

            results = self.yolo_model(
                frame,
                verbose=False,
                conf=0.15,
                classes=[0],
                imgsz=1280
            )

            player_detections = []

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    confidence = float(box.conf[0])

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    x1 = max(0, min(int(x1), width - 1))
                    y1 = max(0, min(int(y1), height - 1))
                    x2 = max(0, min(int(x2), width - 1))
                    y2 = max(0, min(int(y2), height - 1))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    player_detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(confidence, 4),
                        "center": [
                            int((x1 + x2) / 2),
                            int((y1 + y2) / 2)
                        ]
                    })

            print(
                f"PLAYER DETECTION: {len(player_detections)} players detected",
                flush=True
            )

            return player_detections

        except Exception as e:
            print(f"PLAYER DETECTION ERROR: {e}", flush=True)
            return []
        
    def track_players(self, detections, frame):
        """Track detected players."""
        tracked_players = []

        for i, detection in enumerate(detections):
            tracked_players.append({
                "id": f"player_{i}",
                "bbox": detection["bbox"],
                "center": detection["center"],
                "confidence": detection["confidence"]
            })

        return tracked_players
    
    def track_ball(self, detection, frame):
        """Track ball across frames"""
        if detection is None:
            return None
        
        return {
            'bbox': detection['bbox'],
            'center': detection['center'],
            'area': detection['area']
        }
    
    def draw_overlays(self, frame, players, ball):
        """Draw tracking overlays on frame"""
        # Draw player bounding boxes and IDs
        for player in players:
            x1, y1, x2, y2 = player['bbox']
            center = player['center']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw player ID
            cv2.putText(frame, player['id'], (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(frame, tuple(center), 3, (255, 0, 0), -1)
        
        # Draw ball
        if ball:
            x1, y1, x2, y2 = ball['bbox']
            center = ball['center']
            
            # Draw ball bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw ball center
            cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def frame_to_base64(self, frame):
        """Convert frame to base64 for transmission"""
        try:
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            # Convert to base64
            import base64
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Error converting frame to base64: {e}")
            return ""
    
    def get_tracking_data(self):
        """Get all tracking data"""
        return self.tracking_data
    
    def get_analysis_results(self):
        """Get comprehensive analysis results"""
        return {
            'session_id': self.session_id,
            'sport': self.sport,
            'total_frames': len(self.tracking_data['frames']),
            'duration': self.tracking_data['timestamps'][-1] - self.tracking_data['timestamps'][0] if self.tracking_data['timestamps'] else 0,
            'average_fps': self.current_fps,
            'player_detections': len([f for f in self.tracking_data['frames'] if f['players']]),
            'ball_detections': len([f for f in self.tracking_data['frames'] if f['ball']])
        }
    
    def get_status(self):
        """Get current processing status"""
        return {
            'is_processing': self.is_processing,
            'is_live': self.is_live,
            'session_id': self.session_id,
            'sport': self.sport,
            'created_at': self.created_at.isoformat()
        }
    
    def stop(self):
        """Stop video processing"""
        self.is_processing = False
        if self.cap:
            self.cap.release()
        print(f"Stopped processing for session {self.session_id}") 