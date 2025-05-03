import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

def analyze_pose(keypoints):
    keypoints_array = np.array(keypoints)
    
    # Extract relevant keypoints
    # 0: nose, 5-6: shoulders, 11-12: hips, 15-16: ankles
    shoulders = keypoints_array[[5, 6]]
    hips = keypoints_array[[11, 12]]
    ankles = keypoints_array[[15, 16]]
    
    # Calculate body orientation (angle between shoulders and hips)
    shoulder_center = np.mean(shoulders, axis=0)
    hip_center = np.mean(hips, axis=0)
    
    # Calculate angle (in degrees) between vertical line and torso
    # In normal standing pose, this should be close to 90 degrees
    # For a falling person, this will deviate significantly
    dy = hip_center[1] - shoulder_center[1]
    dx = hip_center[0] - shoulder_center[0]
    
    # Calculate angle with vertical axis (90 degrees is vertical/standing)
    body_angle = 90 - np.degrees(np.arctan2(dy, dx))
    
    # Calculate distance between ankles and hips on Y-axis (vertical distance)
    ankle_y = np.mean(ankles[:, 1])
    hip_y = np.mean(hips[:, 1])
    vertical_distance = ankle_y - hip_y  # Should be positive for standing (ankles below hips)
    
    # Calculate horizontal stability (distance between shoulder and hip centers on X-axis)
    horizontal_alignment = abs(shoulder_center[0] - hip_center[0])
    
    # Calculate distance between shoulders (to normalize measurements based on person size)
    shoulder_width = np.linalg.norm(shoulders[0] - shoulders[1])
    
    # Normalize horizontal alignment relative to shoulder width
    normalized_horizontal_alignment = horizontal_alignment / shoulder_width if shoulder_width > 0 else 0
    
    return {
        "body_angle": body_angle,
        "vertical_distance": vertical_distance,
        "horizontal_alignment": horizontal_alignment,
        "normalized_horizontal_alignment": normalized_horizontal_alignment,
        "shoulder_width": shoulder_width
    }

def is_falling(pose_metrics, threshold_angle=30, threshold_horizontal=0.5):
    # A person is falling if:
    # 1. Their body angle deviates significantly from vertical (90 degrees)
    # 2. Their horizontal alignment is significant relative to their shoulder width
    
    # Check if the body angle deviates significantly from vertical
    angle_deviation = abs(pose_metrics["body_angle"])
    is_tilted = angle_deviation > threshold_angle
    
    # Check if horizontal misalignment is significant relative to shoulder width
    horizontal_instability = pose_metrics["normalized_horizontal_alignment"] > threshold_horizontal
    
    # Determine fall state based on combined conditions
    if is_tilted and horizontal_instability:
        return True, f"Person is falling - unstable posture detected (angle: {angle_deviation:.1f}°)"
    elif is_tilted:
        return True, f"Person is falling - tilted orientation detected (angle: {angle_deviation:.1f}°)"
    elif horizontal_instability:
        return False, f"Person may be unstable but not falling (normalized alignment: {pose_metrics['normalized_horizontal_alignment']:.2f})"
    else:
        return False, "Person is in stable position"

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