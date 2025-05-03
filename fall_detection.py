import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from collections import deque
import time

class PoseHistory:
    def __init__(self, max_history=10):
        self.history = {}
        self.max_history = max_history
        self.last_update = {}
        self.cleanup_interval = 5.0
        self.last_cleanup = time.time()
    
    def add(self, track_id, pose_metrics):
        current_time = time.time()
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.max_history)
            self.last_update[track_id] = current_time
        self.history[track_id].append(pose_metrics)
        self.last_update[track_id] = current_time
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_stale_tracks(current_time)
            self.last_cleanup = current_time
    
    def get(self, track_id):
        return list(self.history.get(track_id, []))
    
    def _cleanup_stale_tracks(self, current_time, stale_threshold=10.0):
        stale_tracks = []
        for track_id, last_time in self.last_update.items():
            if current_time - last_time > stale_threshold:
                stale_tracks.append(track_id)
        for track_id in stale_tracks:
            del self.history[track_id]
            del self.last_update[track_id]

def analyze_pose(keypoints, confidence_scores=None):
    keypoints_array = np.array(keypoints)
    nose = keypoints_array[0]
    shoulders = keypoints_array[[5, 6]]
    hips = keypoints_array[[11, 12]]
    ankles = keypoints_array[[15, 16]]
    valid_shoulders = shoulders
    valid_hips = hips
    valid_ankles = ankles
    if confidence_scores is not None:
        confidence_array = np.array(confidence_scores)
        threshold = 0.5
        shoulder_conf = confidence_array[[5, 6]]
        hip_conf = confidence_array[[11, 12]]
        ankle_conf = confidence_array[[15, 16]]
        shoulder_valid = shoulder_conf > threshold
        hip_valid = hip_conf > threshold
        ankle_valid = ankle_conf > threshold
        if np.any(shoulder_valid):
            valid_shoulders = shoulders[shoulder_valid]
        if np.any(hip_valid):
            valid_hips = hips[hip_valid]
        if np.any(ankle_valid):
            valid_ankles = ankles[ankle_valid]
    shoulder_center = np.mean(valid_shoulders, axis=0) if len(valid_shoulders) > 0 else shoulders[0]
    hip_center = np.mean(valid_hips, axis=0) if len(valid_hips) > 0 else hips[0]
    ankle_center = np.mean(valid_ankles, axis=0) if len(valid_ankles) > 0 else ankles[0]
    
    # Fix for the boolean indexing issue
    interest_points = np.array([5, 6, 11, 12, 15, 16])
    mask = np.zeros(len(keypoints_array), dtype=bool)
    for idx in interest_points:
        if idx < len(keypoints_array):
            mask[idx] = True
    all_valid_points = keypoints_array[mask]
    
    if len(all_valid_points) > 0:
        x_min, y_min = np.min(all_valid_points, axis=0)
        x_max, y_max = np.max(all_valid_points, axis=0)
        person_height = y_max - y_min
        person_width = x_max - x_min
    else:
        person_height = np.max(keypoints_array[:, 1]) - np.min(keypoints_array[:, 1])
        person_width = np.max(keypoints_array[:, 0]) - np.min(keypoints_array[:, 0])
    height_width_ratio = person_height / person_width if person_width > 0 else 0
    dy = hip_center[1] - shoulder_center[1]
    dx = hip_center[0] - shoulder_center[0]
    body_angle = np.degrees(np.arctan2(dx, dy))
    vertical_distance = ankle_center[1] - hip_center[1]
    horizontal_alignment = abs(shoulder_center[0] - hip_center[0])
    shoulder_width = np.linalg.norm(shoulders[0] - shoulders[1]) if len(shoulders) >= 2 else person_width / 2
    normalized_horizontal_alignment = horizontal_alignment / shoulder_width if shoulder_width > 0 else 0
    normalized_vertical_distance = vertical_distance / person_height if person_height > 0 else 0
    return {
        "body_angle": body_angle,
        "vertical_distance": vertical_distance,
        "horizontal_alignment": horizontal_alignment,
        "normalized_horizontal_alignment": normalized_horizontal_alignment,
        "normalized_vertical_distance": normalized_vertical_distance,
        "shoulder_width": shoulder_width,
        "person_height": person_height,
        "person_width": person_width,
        "height_width_ratio": height_width_ratio,
        "timestamp": time.time()
    }

def calculate_temporal_features(current_metrics, history):
    if not history or len(history) < 2:
        return {
            "angle_velocity": 0,
            "angle_acceleration": 0,
            "height_ratio_velocity": 0
        }
    current_time = current_metrics["timestamp"]
    prev_time = history[-1]["timestamp"]
    prev_prev_time = history[-2]["timestamp"] if len(history) >= 3 else prev_time
    dt1 = current_time - prev_time
    dt2 = prev_time - prev_prev_time
    if dt1 == 0:
        dt1 = 0.033
    if dt2 == 0:
        dt2 = 0.033
    angle_velocity = (current_metrics["body_angle"] - history[-1]["body_angle"]) / dt1
    height_ratio_velocity = (current_metrics["height_width_ratio"] - history[-1]["height_width_ratio"]) / dt1
    prev_angle_velocity = (history[-1]["body_angle"] - history[-2]["body_angle"]) / dt2 if len(history) >= 3 else 0
    angle_acceleration = (angle_velocity - prev_angle_velocity) / dt1
    return {
        "angle_velocity": angle_velocity,
        "angle_acceleration": angle_acceleration,
        "height_ratio_velocity": height_ratio_velocity
    }

def analyze_statistical_stability(history, window=5):
    if not history or len(history) < window:
        return {
            "angle_stability": 1.0,
            "position_stability": 1.0
        }
    recent = history[-window:]
    angles = [m["body_angle"] for m in recent]
    horizontal_alignments = [m["normalized_horizontal_alignment"] for m in recent]
    angle_stability = 1.0 / (np.std(angles) + 1.0)
    position_stability = 1.0 / (np.std(horizontal_alignments) + 1.0)
    return {
        "angle_stability": angle_stability,
        "position_stability": position_stability
    }

def is_distant_person(pose_metrics, image_width=None, image_height=None):
    if pose_metrics["shoulder_width"] < 30:
        return True
    if image_width and image_height:
        image_area = image_width * image_height
        person_area = pose_metrics["person_width"] * pose_metrics["person_height"]
        person_ratio = person_area / image_area
        if person_ratio < 0.05:
            return True
    return False

def is_falling(pose_metrics, track_id=None, pose_history=None, image_size=None, 
               threshold_angle=30, threshold_horizontal=0.5):
    history = []
    if pose_history is not None and track_id is not None:
        pose_history.add(track_id, pose_metrics)
        history = pose_history.get(track_id)
    image_width, image_height = image_size if image_size else (None, None)
    is_distant = is_distant_person(pose_metrics, image_width, image_height)
    if is_distant:
        adjusted_angle_threshold = threshold_angle * 0.7
        adjusted_horizontal_threshold = threshold_horizontal * 0.8
    else:
        adjusted_angle_threshold = threshold_angle
        adjusted_horizontal_threshold = threshold_horizontal
    angle_deviation = abs(pose_metrics["body_angle"])
    is_tilted = angle_deviation > adjusted_angle_threshold
    horizontal_instability = pose_metrics["normalized_horizontal_alignment"] > adjusted_horizontal_threshold
    is_lying_down = pose_metrics["height_width_ratio"] < 1.2
    static_confidence = 0.0
    if is_tilted:
        angle_factor = min((angle_deviation - adjusted_angle_threshold) / 60.0, 1.0)
        static_confidence += 0.5 * angle_factor
    if horizontal_instability:
        horiz_factor = min((pose_metrics["normalized_horizontal_alignment"] - adjusted_horizontal_threshold) / 0.5, 1.0)
        static_confidence += 0.3 * horiz_factor
    if is_lying_down:
        ratio_factor = min((1.2 - pose_metrics["height_width_ratio"]) / 0.7, 1.0) if pose_metrics["height_width_ratio"] > 0 else 0
        static_confidence += 0.2 * ratio_factor
    temporal_confidence = 0.0
    if len(history) >= 3:
        temporal_features = calculate_temporal_features(pose_metrics, history)
        stability_features = analyze_statistical_stability(history)
        sudden_angle_change = abs(temporal_features["angle_velocity"]) > 45
        sudden_acceleration = abs(temporal_features["angle_acceleration"]) > 90
        sudden_height_ratio_change = abs(temporal_features["height_ratio_velocity"]) > 0.8
        unstable_posture = stability_features["angle_stability"] < 0.2
        if sudden_angle_change:
            temporal_confidence += 0.3 * min(abs(temporal_features["angle_velocity"]) / 90.0, 1.0)
        if sudden_acceleration:
            temporal_confidence += 0.2 * min(abs(temporal_features["angle_acceleration"]) / 180.0, 1.0)
        if sudden_height_ratio_change:
            temporal_confidence += 0.2 * min(abs(temporal_features["height_ratio_velocity"]) / 1.6, 1.0)
        if unstable_posture:
            temporal_confidence += 0.3 * (1.0 - stability_features["angle_stability"] * 5)
    if len(history) >= 3:
        final_confidence = 0.4 * static_confidence + 0.6 * temporal_confidence
    else:
        final_confidence = static_confidence
    is_fall = final_confidence > 0.5
    if is_fall:
        if final_confidence > 0.8:
            message = f"HIGH CONFIDENCE FALL DETECTED (score: {final_confidence:.2f})"
        else:
            message = f"Fall detected (confidence: {final_confidence:.2f})"
        details = []
        if is_tilted:
            details.append(f"body tilt: {angle_deviation:.1f}°")
        if horizontal_instability:
            details.append(f"horizontal instability: {pose_metrics['normalized_horizontal_alignment']:.2f}")
        if is_lying_down:
            details.append(f"lying posture: {pose_metrics['height_width_ratio']:.2f}")
        if len(history) >= 3 and sudden_angle_change:
            details.append(f"rapid movement: {temporal_features['angle_velocity']:.1f}°/s")
        if details:
            message += " - " + ", ".join(details)
    else:
        if final_confidence > 0.3:
            message = f"Unusual posture detected (score: {final_confidence:.2f}), but not classified as fall"
        else:
            message = "Person is in stable position"
    return is_fall, message, final_confidence


def draw_fall_detection(image_path, pose_data, bboxes_data, output_path):
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    
    # Get keypoints and edges
    keypoints_list = pose_data["keypoints"]
    edges = pose_data["edges"]
    
    # Get bounding boxes
    bboxes = np.array(bboxes_data["bboxes"])
    
    # Process each person
    fall_detection_results = []
    
    for i, keypoints in enumerate(keypoints_list):
        # Analyze pose
        pose_metrics = analyze_pose(keypoints)
        is_fall, message = is_falling(pose_metrics)
        
        # Add fall detection results
        result = {
            "person_id": i,
            "pose_metrics": pose_metrics,
            "is_falling": is_fall,
            "message": message
        }
        fall_detection_results.append(result)
        
        # Add text to image based on fall detection
        status_text = "FALLING" if is_fall else "NORMAL"
        color = (0, 0, 255) if is_fall else (0, 255, 0)
        
        # Get bounding box if available
        if i < len(bboxes):
            bbox = bboxes[i]
            x1, y1, x2, y2 = [int(b) for b in bbox]
            
            cv2.putText(
                image,
                status_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )
            
            # Add rectangle with fall status color
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2
            )
            
            # Draw angle information
            angle_text = f"Angle: {abs(pose_metrics['body_angle']):.1f}°"
            cv2.putText(
                image,
                angle_text,
                (x1, y1 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
    
    # Save the image with fall detection annotations
    cv2.imwrite(output_path, image)
    
    return fall_detection_results

def main():
    parser = argparse.ArgumentParser(description="Fall detection using pose estimation data")
    parser.add_argument("--image", type=str, required=True, help="Path to original/pose image")
    parser.add_argument("--keypoints", type=str, help="Path to keypoints JSON file (default: outputs/[image_name]_pose_keypoints.json)")
    parser.add_argument("--bboxes", type=str, help="Path to bounding boxes JSON file (default: outputs/[image_name]_pose_bboxes.json)")
    parser.add_argument("--output", type=str, help="Path to output image (default: outputs/[image_name]_fall_detection.jpg)")
    args = parser.parse_args()
    
    # Set up paths
    image_path = args.image
    image_name = os.path.basename(image_path).split('.')[0]
    
    keypoints_path = args.keypoints if args.keypoints else f"outputs/{image_name}_pose_keypoints.json"
    bboxes_path = args.bboxes if args.bboxes else f"outputs/{image_name}_pose_bboxes.json"
    output_path = args.output if args.output else f"outputs/{image_name}_fall_detection.jpg"
    
    # Create outputs directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Load keypoints and bounding boxes
    try:
        with open(keypoints_path, 'r') as f:
            pose_data = json.load(f)
        
        with open(bboxes_path, 'r') as f:
            bboxes_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure to run infer_img.py first to generate the keypoints and bounding boxes files.")
        return
    
    print(f"Processing image: {image_path}")
    print(f"Using keypoints from: {keypoints_path}")
    print(f"Using bounding boxes from: {bboxes_path}")
    
    # Process the image
    results = draw_fall_detection(image_path, pose_data, bboxes_data, output_path)
    
    # Save fall detection results
    results_path = f"outputs/{image_name}_fall_detection_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Display results
    print("\nFall Detection Results:")
    for i, result in enumerate(results):
        print(f"Person {i}: {result['message']}")
        print(f"  - Body angle: {result['pose_metrics']['body_angle']:.2f} degrees")
        print(f"  - Normalized horizontal alignment: {result['pose_metrics']['normalized_horizontal_alignment']:.2f}")
        print(f"  - Shoulder width: {result['pose_metrics']['shoulder_width']:.2f}")
    
    print(f"\nFall detection image saved to: {output_path}")
    print(f"Fall detection results saved to: {results_path}")
    
    # Display the image
    try:
        result_image = cv2.imread(output_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Fall Detection Results")
        plt.show()
    except Exception as e:
        print(f"Unable to display result image: {e}")

if __name__ == "__main__":
    main()