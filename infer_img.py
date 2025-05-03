import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
import os
import json
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
import time
import argparse

edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6), (11, 12)
]

def detect_objects(image, model_detection, image_processor_detection, device):
    inputs = image_processor_detection(images=image, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model_detection(**inputs)

    target_sizes = torch.tensor([(image.height, image.width)])
    
    results = image_processor_detection.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.8
    )
    
    result = results[0]

    return result

def draw_bbox(image, boxes_xyxy):
    img_with_boxes = np.array(image.copy())
    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
    
    for box in boxes_xyxy:
        x1, y1, x2, y2 = [int(b) for b in box]
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return img_with_boxes

def detect_pose(image, person_boxes, model_pose, image_processor_pose, device):
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
    image_pose_result = pose_results[0]

    return image_pose_result

def draw_keypoints(outputs, image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    keypoints_data = []
    for i, pose_result in enumerate(outputs):
        # Move keypoints to CPU before converting to numpy
        keypoints = pose_result['keypoints'].cpu().detach().numpy()
        keypoints = keypoints[:, :].reshape(-1, 2)
        keypoints_data.append(keypoints.tolist())
        
        for p in range(keypoints.shape[0]):
            cv2.circle(
                image, 
                (int(keypoints[p, 0]), int(keypoints[p, 1])), 
                3, (0, 0, 255), 
                thickness=-1, 
                lineType=cv2.FILLED
            )
            cv2.putText(
                image, 
                f'{p}', 
                (int(keypoints[p, 0]+10), int(keypoints[p, 1]-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                1
            )
            
        for ie, e in enumerate(edges):
            rgb = matplotlib.colors.hsv_to_rgb([ie/float(len(edges)), 1.0, 1.0])
            rgb = rgb*255
            
            cv2.line(
                image, 
                (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                tuple(rgb), 
                2, 
                lineType=cv2.LINE_AA
            )
    return image, keypoints_data

def main():
    parser = argparse.ArgumentParser(description="Object detection and pose estimation")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs('outputs', exist_ok=True)
    
    image_path = args.image
    image_name = os.path.basename(image_path).split('.')[0]
    image = Image.open(image_path)
    
    print("Loading models...")
    image_processor_detection = AutoProcessor.from_pretrained('PekingU/rtdetr_r50vd_coco_o365')
    image_processor_pose = AutoProcessor.from_pretrained('usyd-community/vitpose-base-simple')
    
    model_detection = RTDetrForObjectDetection.from_pretrained(
        'PekingU/rtdetr_r50vd_coco_o365', 
        device_map=device
    )
    
    model_pose = VitPoseForPoseEstimation.from_pretrained(
        'usyd-community/vitpose-base-simple', 
        device_map=device
    )
    
    print("Detecting objects...")
    start = time.time()
    result = detect_objects(image, model_detection, image_processor_detection, device)
    print("Result:", result)
    end = time.time()
    print(f"Object detection time: {end - start:.2f} seconds")
    
    img_with_boxes = draw_bbox(image, result['boxes'])
    detection_path = f'outputs/{image_name}_detection.jpg'
    cv2.imwrite(detection_path, img_with_boxes)
    print(f"Detection image saved to {detection_path}")
    
    if len(result['boxes']) == 0:
        print("No persons detected in the image.")
        return
    
    print("Estimating poses...")
    start = time.time()
    image_pose_result = detect_pose(image, result['boxes'], model_pose, image_processor_pose, device)
    end = time.time()
    print(f"Pose estimation time: {end - start:.2f} seconds")
    
    # Save bounding boxes to file
    bboxes_path = f'outputs/{image_name}_pose_bboxes.json'
    with open(bboxes_path, 'w') as f:
        json.dump({
            'bboxes': result['boxes'].cpu().tolist(),  # Move to CPU before converting to list
        }, f)
    print(f"Bounding boxes saved to {bboxes_path}")
    
    # Draw keypoints and save to file
    image_with_keypoints, keypoints_data = draw_keypoints(image_pose_result, image)
    pose_image_path = f'outputs/{image_name}_pose.jpg'
    cv2.imwrite(pose_image_path, image_with_keypoints)
    print(f"Pose image saved to {pose_image_path}")
    
    # Save keypoints to file
    keypoints_path = f'outputs/{image_name}_pose_keypoints.json'
    with open(keypoints_path, 'w') as f:
        json.dump({
            'keypoints': keypoints_data,
            'edges': edges
        }, f)
    print(f"Keypoints saved to {keypoints_path}")

if __name__ == "__main__":
    main()