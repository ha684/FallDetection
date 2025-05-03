import torch
import numpy as np
import cv2
import os
import json
import time
import argparse
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
from collections import defaultdict
from fall_detection import analyze_pose, is_falling, PoseHistory
from deep_sort_realtime.deepsort_tracker import DeepSort

def detect_objects(image, model_detection, image_processor_detection, device):
    """Detect objects (people) in an image"""
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    inputs = image_processor_detection(images=image, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model_detection(**inputs)

    target_sizes = torch.tensor([(image.height, image.width)])
    
    results = image_processor_detection.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.7
    )
    
    result = results[0]
    
    # Filter for people (class 0)
    people_indices = [i for i, label in enumerate(result['labels']) if label == 0]
    people_boxes = result['boxes'][people_indices]
    people_scores = result['scores'][people_indices]
    
    return {
        'boxes': people_boxes,
        'scores': people_scores,
        'labels': torch.zeros(len(people_indices), dtype=torch.int32).to(device)  # All labels are 0 (person)
    }

def detect_pose(image, person_boxes, model_pose, image_processor_pose, device):
    """Detect pose for each person in the image"""
    if len(person_boxes) == 0:
        return []
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Move boxes to CPU before passing them to the image processor
    boxes_cpu = person_boxes.cpu()
    
    inputs = image_processor_pose(
        image, boxes=[boxes_cpu], return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = model_pose(**inputs)
    
    pose_results = image_processor_pose.post_process_pose_estimation(
        outputs, boxes=[boxes_cpu]
    )
    
    return pose_results[0]

def process_video(video_path, output_path, model_detection, model_pose, 
                  image_processor_detection, image_processor_pose, device,
                  frame_interval=1, max_frames=None, display=True):
    """Process a video for fall detection with DeepSORT tracking"""
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize DeepSORT tracker
    tracker = DeepSort(
        max_age=30,                    
        n_init=3,                 
        nms_max_overlap=0.8,     
        max_cosine_distance=0.3,
        nn_budget=100
    )
    
    # Initialize variables
    frame_count = 0
    processed_count = 0
    fall_timestamps = []
    person_fall_status = defaultdict(lambda: {'is_falling': False, 'last_detected': 0})
    last_tracks = []
    fall_ids = set()    # Track IDs that have already triggered fall alerts
    
    # Initialize pose history for temporal analysis
    pose_history = PoseHistory(max_history=10)
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    print(f"Processing every {frame_interval} frame(s)")
    
    start_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Time in seconds
            
            # Determine if we should process this frame for detection
            should_process = (frame_count % frame_interval == 0)
            
            # Dictionary to store pose results by track_id
            tracked_pose_results = {}
            
            if should_process:
                processed_count += 1
                if max_frames is not None and processed_count > max_frames:
                    print(f"Reached maximum frames limit ({max_frames})")
                    break
                
                # Print progress
                progress = frame_count / total_frames * 100
                elapsed = time.time() - start_time
                remaining = (elapsed / processed_count) * ((total_frames / frame_interval) - processed_count) if processed_count > 0 else 0
                print(f"\rProcessing frame {frame_count}/{total_frames} ({progress:.1f}%) - "
                      f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s", end="")
                
                # Convert frame to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Detect people
                detection_result = detect_objects(pil_image, model_detection, image_processor_detection, device)
                
                if len(detection_result['boxes']) > 0:
                    # Format detections for DeepSORT
                    boxes = detection_result['boxes'].cpu().numpy()
                    scores = detection_result['scores'].cpu().numpy()
                    
                    # Prepare detections in format expected by DeepSORT
                    # DeepSORT expects detections as a list of lists, each containing:
                    # [left, top, right, bottom, confidence]
                    deepsort_detections = []
                    for i, (box, score) in enumerate(zip(boxes, scores)):
                        x1, y1, x2, y2 = box
                        # DeepSORT expects [left, top, width, height, confidence] format
                        # But DeepSort-Realtime uses [left, top, right, bottom, confidence]
                        deepsort_detections.append(([float(x1), float(y1), float(x2), float(y2)], float(score), None))
                    
                    # Update tracker with new detections
                    tracks = tracker.update_tracks(deepsort_detections, frame=frame)
                    last_tracks = [t for t in tracks if t.is_confirmed()]
                    
                    # Get pose estimations for each tracked person
                    for track in last_tracks:
                        if not track.is_confirmed():
                            continue
                            
                        track_id = track.track_id
                        bbox = track.to_ltrb()  # Left, Top, Right, Bottom format
                        
                        # Convert bbox back to torch tensor for pose estimation
                        person_box = torch.tensor([bbox], device=device)
                        
                        # Detect pose for this person
                        pose_result = detect_pose(pil_image, person_box, model_pose, image_processor_pose, device)
                        if pose_result:
                            tracked_pose_results[track_id] = {
                                'pose': pose_result[0],
                                'bbox': bbox
                            }
                            
                            # Analyze pose for fall detection
                            keypoints = pose_result[0]['keypoints'].cpu().detach().numpy().reshape(-1, 2).tolist()
                            
                            # Get confidence scores if available
                            confidence_scores = None
                            if 'scores' in pose_result[0]:
                                confidence_scores = pose_result[0]['scores'].cpu().detach().numpy().tolist()
                                
                            # Analyze pose metrics
                            pose_metrics = analyze_pose(keypoints, confidence_scores)
                            
                            # Pass image size for distance estimation
                            image_size = (width, height)
                            
                            # Use enhanced fall detection with temporal analysis
                            is_fall, message, confidence = is_falling(
                                pose_metrics, 
                                track_id=track_id, 
                                pose_history=pose_history,
                                image_size=image_size
                            )
                            
                            # Update person fall status
                            person_fall_status[track_id]['is_falling'] = is_fall
                            person_fall_status[track_id]['last_detected'] = frame_count
                            person_fall_status[track_id]['pose_metrics'] = pose_metrics
                            person_fall_status[track_id]['message'] = message
                            person_fall_status[track_id]['confidence'] = confidence
                            
                            # Record new fall events (that weren't already recorded)
                            if is_fall and track_id not in fall_ids:
                                fall_ids.add(track_id)
                                fall_timestamps.append({
                                    'time': current_time,
                                    'frame': frame_count,
                                    'track_id': track_id,
                                    'confidence': confidence,
                                    'message': message
                                })
                                print(f"\nFall detected for person {track_id} at {current_time:.2f}s (frame {frame_count}) - Confidence: {confidence:.2f}")
            
            # Create a copy of the frame to draw on
            annotated_frame = frame.copy()
            
            # Use the last available tracks for drawing
            if last_tracks:
                for track in last_tracks:
                    track_id = track.track_id
                    bbox = track.to_ltrb()  # Get bounding box in Left, Top, Right, Bottom format
                    
                    # Only draw if the track was recently detected (within 2*interval frames)
                    if track_id in person_fall_status and frame_count - person_fall_status[track_id]['last_detected'] <= 2 * frame_interval:
                        x1, y1, x2, y2 = [int(b) for b in bbox]
                        
                        # Determine color based on fall status
                        is_fall = person_fall_status[track_id].get('is_falling', False)
                        confidence = person_fall_status[track_id].get('confidence', 0)
                        
                        # Adjust color intensity based on confidence
                        if is_fall:
                            # Red with intensity based on confidence
                            color = (0, 0, min(255, int(255 * confidence)))
                        else:
                            # Green
                            color = (0, 255, 0)
                        
                        # Draw bounding box with ID
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add text for status and ID
                        if is_fall:
                            status_text = f"ID:{track_id} - FALLING ({confidence:.2f})"
                        else:
                            status_text = f"ID:{track_id} - NORMAL"
                            
                        cv2.putText(
                            annotated_frame,
                            status_text,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2
                        )
                        
                        # Add angle information if available
                        if 'pose_metrics' in person_fall_status[track_id]:
                            angle_text = f"Angle: {abs(person_fall_status[track_id]['pose_metrics']['body_angle']):.1f}°"
                            cv2.putText(
                                annotated_frame,
                                angle_text,
                                (x1, y1 - 35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2
                            )
            
            # Draw overall fall alert if any recent falls
            active_falls = [k for k, v in person_fall_status.items() 
                          if v['is_falling'] and frame_count - v['last_detected'] <= 2 * frame_interval]
            
            if active_falls:
                # Calculate average confidence for the alert
                avg_confidence = sum(person_fall_status[id]['confidence'] for id in active_falls) / len(active_falls)
                
                alert_text = f"FALL DETECTED! ({len(active_falls)} person(s)) - Confidence: {avg_confidence:.2f}"
                cv2.putText(
                    annotated_frame,
                    alert_text,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3
                )
            
            # Write the annotated frame to output video
            out.write(annotated_frame)
            
            # Display frame if requested
            if display:
                cv2.imshow('Fall Detection', annotated_frame)
                
                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except Exception as e:
        print(f"\nError processing frame {frame_count}: {e}")
        import traceback
        traceback.print_exc()  # Print detailed error information
    
    finally:
        # Release resources
        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()
        
        # Save fall timestamps
        video_name = os.path.basename(video_path).split('.')[0]
        timestamps_path = f"outputs/{video_name}_fall_timestamps.json"
        os.makedirs(os.path.dirname(timestamps_path), exist_ok=True)
        with open(timestamps_path, 'w') as f:
            json.dump({
                "video_path": video_path,
                "total_frames": total_frames,
                "fps": fps,
                "fall_timestamps": fall_timestamps
            }, f, indent=4)
            
        print(f"\nProcessed {processed_count} frames in {time.time() - start_time:.2f} seconds")
        print(f"Output video saved to: {output_path}")
        print(f"Fall timestamps saved to: {timestamps_path}")
        
        if fall_timestamps:
            print(f"Falls detected: {len(fall_timestamps)} events")
            for fall in fall_timestamps:
                print(f"  Person {fall['track_id']} at {fall['time']:.2f}s (frame {fall['frame']}) - Confidence: {fall['confidence']:.2f}")
                print(f"    {fall['message']}")
        else:
            print("No falls detected.")

def main():
    parser = argparse.ArgumentParser(description="Fall detection in video using pose estimation and DeepSORT tracking")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, help="Path to output video (default: outputs/<video_name>_fall_detection.mp4)")
    parser.add_argument("--interval", type=int, default=5, help="Process every Nth frame (default: 5)")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to process")
    parser.add_argument("--no-display", action="store_true", help="Disable displaying video while processing")
    args = parser.parse_args()
    
    # Set up paths
    video_path = args.video
    video_name = os.path.basename(video_path).split('.')[0]
    output_path = args.output if args.output else f"outputs/{video_name}_fall_detection.mp4"
    
    # Create outputs directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load models
    print("Loading detection model...")
    image_processor_detection = AutoProcessor.from_pretrained('PekingU/rtdetr_r50vd_coco_o365')
    model_detection = RTDetrForObjectDetection.from_pretrained(
        'PekingU/rtdetr_r50vd_coco_o365', 
        device_map=device
    )
    
    print("Loading pose estimation model...")
    image_processor_pose = AutoProcessor.from_pretrained('usyd-community/vitpose-base-simple')
    model_pose = VitPoseForPoseEstimation.from_pretrained(
        'usyd-community/vitpose-base-simple', 
        device_map=device
    )
    
    # Process the video
    process_video(
        video_path, 
        output_path, 
        model_detection, 
        model_pose, 
        image_processor_detection, 
        image_processor_pose, 
        device,
        frame_interval=args.interval,
        max_frames=args.max_frames,
        display=not args.no_display
    )

if __name__ == "__main__":
    main()