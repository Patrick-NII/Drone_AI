import cv2
import os
import yt_dlp
from ultralytics import YOLO
from datetime import datetime
import param
import torch
import numpy as np
from pathlib import Path

def setup_device():
    """Configure the device for optimal performance on macOS."""
    if param.USE_MPS and torch.backends.mps.is_available():
        return "mps"  # Use Metal Performance Shaders for macOS
    return "cpu"  # Fallback to CPU

def get_youtube_stream_url(youtube_url):
    """Extract YouTube stream URL with optimized settings."""
    ydl_opts = {
        "format": "best[ext=mp4]",
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info["url"]
    except Exception as e:
        print(f"Error extracting YouTube stream: {e}")
        return None

def resize_frame(frame):
    """Resize frame while maintaining aspect ratio."""
    height, width = frame.shape[:2]
    if width > param.MAX_FRAME_WIDTH or height > param.MAX_FRAME_HEIGHT:
        scale = min(param.MAX_FRAME_WIDTH/width, param.MAX_FRAME_HEIGHT/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height))
    return frame

def process_detections(frame, results, object_counter):
    """Process and visualize detections on the frame."""
    for det in results[0].boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        cls = int(det.cls[0])
        cls_name = results[0].names[cls]
        
        # Get label from dictionary or use original name
        cls_name = param.OBJECT_LABELS.get(cls_name, cls_name)
        
        # Update object counter
        if cls_name not in object_counter:
            object_counter[cls_name] = 1
        else:
            object_counter[cls_name] += 1
            
        label = f"{cls_name} {object_counter[cls_name]}"
        
        # Get color for the box
        color = param.BOX_COLORS.get(cls_name.lower(), param.DEFAULT_BOX_COLOR)
        
        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, param.TEXT_FONT, 
                   param.TEXT_COLOR, param.TEXT_THICKNESS)
        
        # Save object image if enabled
        if param.SAVE_OBJECT_IMAGES:
            object_img = frame[y1:y2, x1:x2]
            if object_img.size > 0:
                object_img_path = os.path.join(param.OUTPUT_DIR, 
                                             f"{cls_name}_{object_counter[cls_name]}.jpg")
                cv2.imwrite(object_img_path, object_img)
    
    return frame

def main():
    # Create output directory
    os.makedirs(param.OUTPUT_DIR, exist_ok=True)
    
    # Initialize YOLO model with optimized settings
    device = setup_device()
    model = YOLO(param.YOLO_MODEL)
    
    # Source selection
    print("\nSelect video source:")
    print("[1] Webcam")
    print("[2] YouTube Video")
    print("[3] Local Video")
    print("[4] Single Image or Image Directory")
    
    choice = input("Enter the corresponding number: ")
    
    if choice == "1":
        source = param.DEFAULT_SOURCE
    elif choice == "2":
        youtube_url = input("Enter YouTube URL: ")
        source = get_youtube_stream_url(youtube_url)
        if not source:
            print("Error: Could not retrieve YouTube stream.")
            return
    elif choice == "3":
        source = input("Enter local video path: ")
        if not os.path.isfile(source):
            print("Error: Video file not found.")
            return
    elif choice == "4":
        source = input("Enter image file or directory path: ")
        if not os.path.exists(source):
            print("Error: File or directory not found.")
            return
    else:
        print("Invalid choice, using default webcam.")
        source = param.DEFAULT_SOURCE
    
    print(f"Selected source: {source}")
    
    # Process based on source type
    if isinstance(source, str) and os.path.isfile(source) and source.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Single image processing
        output_image_path = os.path.join(param.OUTPUT_DIR, 
                                       f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        results = model.predict(source=source, save=True, 
                              show=param.DISPLAY_REAL_TIME,
                              conf=param.CONFIDENCE_THRESHOLD,
                              iou=param.IOU_THRESHOLD,
                              device=device)
        results[0].save(output_image_path)
        print(f"Processed image saved to: {output_image_path}")
        
    elif isinstance(source, str) and os.path.isdir(source):
        # Directory of images processing
        model.predict(source=source, save=True,
                     show=param.DISPLAY_REAL_TIME,
                     conf=param.CONFIDENCE_THRESHOLD,
                     iou=param.IOU_THRESHOLD,
                     device=device)
        print(f"Processed images saved to: {param.OUTPUT_DIR}/")
        
    else:
        # Video processing (Webcam or video file)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return
            
        # Video writer setup
        writer = None
        output_video_path = os.path.join(param.OUTPUT_DIR, "output_video.mp4")
        
        if param.SAVE_PROCESSED_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Initialize object counter
        object_counter = {}
        frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Process every nth frame for better performance
            if frame_count % param.PROCESS_EVERY_N_FRAMES == 0:
                # Resize frame if needed
                frame = resize_frame(frame)
                
                # Run YOLO tracking
                results = model.track(frame, persist=True,
                                   conf=param.CONFIDENCE_THRESHOLD,
                                   iou=param.IOU_THRESHOLD,
                                   tracker=param.TRACKER,
                                   device=device,
                                   half=True)
                
                # Process and visualize detections
                frame = process_detections(frame, results, object_counter)
                
                # Save processed frame
                if param.SAVE_PROCESSED_VIDEO:
                    writer.write(frame)
                
                # Display frame if enabled
                if param.DISPLAY_REAL_TIME:
                    cv2.imshow("YOLOv8 Tracking", frame)
                    
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        if param.SAVE_PROCESSED_VIDEO:
            writer.release()
        cv2.destroyAllWindows()
        print(f"Processed video saved to: {output_video_path}")
    
    print("Processing completed!")

if __name__ == "__main__":
    main()
