import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image
import os
import time
import argparse
from pathlib import Path
import glob

class DetectionPipeline:
    def __init__(self, yolo_path, dish_path, tray_path, device='auto'):
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load models
        self.yolo = YOLO(yolo_path)
        self.dish_model = self._load_classifier(dish_path)
        self.tray_model = self._load_classifier(tray_path)
        
        # Setup
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.classes = ['empty', 'kakigori', 'not_empty']
        self.colors = {'dish': (0, 255, 0), 'tray': (255, 0, 0)}
        self.status_colors = {'empty': (128, 128, 128), 'kakigori': (0, 255, 255), 'not_empty': (0, 0, 255)}
        
        # Sliding window params
        self.window_size = 640
        self.overlap = 0.2
        self.stride = int(self.window_size * (1 - self.overlap))
    
    def _load_classifier(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        from torchvision.models import efficientnet_b1
        model = efficientnet_b1(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.eval().to(self.device)
    
    def enhance_image(self, img):
        # CLAHE enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        return cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
    
    def upscale_image(self, img, target_size=1920):
        h, w = img.shape[:2]
        if max(w, h) < target_size:
            scale = target_size / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            return cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4), scale, scale
        return img, 1.0, 1.0
    
    def preprocess_image(self, img, mode='auto'):
        h, w = img.shape[:2]
        needs_enhancement = mode == 'always' or (mode == 'auto' and max(w, h) < 800)
        
        if mode == 'never':
            return img, 1.0, 1.0
        
        # Upscale if small
        img, scale_x, scale_y = self.upscale_image(img)
        
        # Enhance quality if needed
        if needs_enhancement and mode != 'upscale_only':
            img = self.enhance_image(img)
        
        return img, scale_x, scale_y
    
    def classify_crop(self, crop, obj_type):
        model = self.dish_model if obj_type == 'dish' else self.tray_model
        if model is None:
            return 'unknown', 0.0
        
        if isinstance(crop, np.ndarray):
            crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            return self.classes[pred.item()], conf.item()
    
    def get_windows(self, h, w):
        windows = []
        y_steps = max(1, (h - self.window_size) // self.stride + 1)
        x_steps = max(1, (w - self.window_size) // self.stride + 1)
        
        for y in range(y_steps):
            for x in range(x_steps):
                x1, y1 = x * self.stride, y * self.stride
                x2, y2 = min(x1 + self.window_size, w), min(y1 + self.window_size, h)
                if (y2 - y1) >= 320 and (x2 - x1) >= 320:
                    windows.append((x1, y1, x2, y2))
        return windows
    
    def nms(self, detections, iou_thresh=0.4):
        if not detections:
            return []
        
        detections = sorted(detections, key=lambda x: x.get('det_conf', 0), reverse=True)
        keep = []
        
        while detections:
            current = detections.pop(0)
            keep.append(current)
            detections = [d for d in detections if self._iou(current['bbox'], d['bbox']) < iou_thresh]
        
        return keep
    
    def _iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        return inter / (area1 + area2 - inter)
    
    def detect_objects(self, frame, conf_thresh, use_sliding_window):
        h, w = frame.shape[:2]
        detections = []
        
        if use_sliding_window and (h > self.window_size or w > self.window_size):
            # Sliding window detection
            for x1, y1, x2, y2 in self.get_windows(h, w):
                window = frame[y1:y2, x1:x2]
                if window.shape[:2] != (self.window_size, self.window_size):
                    window = cv2.resize(window, (self.window_size, self.window_size))
                    scale_x, scale_y = (x2-x1)/self.window_size, (y2-y1)/self.window_size
                else:
                    scale_x = scale_y = 1.0
                
                results = self.yolo.predict(window, conf=conf_thresh, device=self.device, verbose=False)[0]
                
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    obj_type = self.yolo.names[cls_id]
                    wx1, wy1, wx2, wy2 = map(int, box.xyxy[0].tolist())
                    
                    # Scale back to frame coordinates
                    fx1 = max(0, min(x1 + int(wx1 * scale_x), w))
                    fy1 = max(0, min(y1 + int(wy1 * scale_y), h))
                    fx2 = max(0, min(x1 + int(wx2 * scale_x), w))
                    fy2 = max(0, min(y1 + int(wy2 * scale_y), h))
                    
                    if fx2 > fx1 and fy2 > fy1:
                        detections.append({
                            'bbox': (fx1, fy1, fx2, fy2),
                            'type': obj_type,
                            'det_conf': float(box.conf[0])
                        })
        else:
            # Single detection
            results = self.yolo.predict(frame, conf=conf_thresh, device=self.device, verbose=False)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                obj_type = self.yolo.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'type': obj_type,
                    'det_conf': float(box.conf[0])
                })
        
        return detections
    
    def process_frame(self, frame, conf_thresh=0.5, use_sliding_window=True, enhance_mode='auto'):
        original = frame.copy()
        
        # Preprocess
        enhanced, scale_x, scale_y = self.preprocess_image(frame, enhance_mode)
        
        # Detect objects
        detections = self.detect_objects(enhanced, conf_thresh, use_sliding_window)
        
        # Scale back to original coordinates
        if scale_x != 1.0 or scale_y != 1.0:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                det['bbox'] = (int(x1/scale_x), int(y1/scale_y), int(x2/scale_x), int(y2/scale_y))
        
        # Apply NMS
        detections = self.nms(detections)
        
        # Classify crops
        final_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(original.shape[1], x2), min(original.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                crop = original[y1:y2, x1:x2]
                status, conf = self.classify_crop(crop, det['type'])
                final_detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'type': det['type'],
                    'status': status,
                    'confidence': conf
                })
        
        return final_detections
    
    def draw_annotations(self, frame, detections):
        result = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            obj_type, status, conf = det['type'], det['status'], det['confidence']
            
            # Colors
            box_color = self.colors.get(obj_type, (255, 255, 255))
            status_color = self.status_colors.get(status, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw label
            text = f"{obj_type}: {status} ({conf:.2f})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result, (x1, y1-th-10), (x1+tw, y1), status_color, -1)
            cv2.putText(result, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        return result
    
    def process_image(self, input_path, output_path, **kwargs):
        print(f"Processing: {input_path}")
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"Error loading: {input_path}")
            return
        
        detections = self.process_frame(frame, **kwargs)
        result = self.draw_annotations(frame, detections)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result)
        
        print(f"Found {len(detections)} objects, saved to: {output_path}")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. {det['type']} - {det['status']} ({det['confidence']:.3f})")
    
    def process_video(self, input_path, output_path, skip_frames=0, **kwargs):
        print(f"Processing video: {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error opening: {input_path}")
            return
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Writer
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = processed = 0
        start_time = time.time()
        last_detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            should_process = skip_frames == 0 or frame_count % (skip_frames + 1) == 1
            
            if should_process:
                detections = self.process_frame(frame, **kwargs)
                last_detections = detections
                processed += 1
                
                if processed % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {processed} frames ({frame_count}/{total_frames}) - {processed/elapsed:.1f} fps")
            else:
                detections = last_detections
            
            result = self.draw_annotations(frame, detections)
            out.write(result)
        
        cap.release()
        out.release()
        
        elapsed = time.time() - start_time
        print(f"Complete: {processed} frames in {elapsed:.1f}s ({processed/elapsed:.1f} fps)")
        print(f"Saved to: {output_path}")


def get_output_path(input_path):
    """Generate output path in test/output with _annotated suffix"""
    input_path = Path(input_path)
    output_dir = Path("test/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_name = input_path.stem + "_annotated" + input_path.suffix
    return str(output_dir / output_name)


def main():
    parser = argparse.ArgumentParser(description="Detection and Classification Pipeline")
    
    # Required
    parser.add_argument('-i', '--input', required=True, help='Input image/video path (supports wildcards)')
    
    # Models
    parser.add_argument('--yolo-model', default='models/detection.pt', help='YOLO model path')
    parser.add_argument('--dish-model', default='models/dish_classifier.pt', help='Dish classifier path')
    parser.add_argument('--tray-model', default='models/tray_classifier.pt', help='Tray classifier path')
    
    # Parameters
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (0-1)')
    parser.add_argument('--skip-frames', type=int, default=30, help='Skip N frames in video')
    parser.add_argument('--enhance', default='auto', choices=['auto', 'always', 'never', 'upscale_only'], 
                       help='Enhancement mode')
    parser.add_argument('--no-sliding-window', action='store_true', help='Disable sliding window')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'], help='Device')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Find input files
    input_files = glob.glob(args.input) if '*' in args.input else [args.input]
    if not input_files:
        print(f"No files found: {args.input}")
        return 1
    
    # Validate model files
    for name, path in [('YOLO', args.yolo_model), ('Dish', args.dish_model), ('Tray', args.tray_model)]:
        if not Path(path).exists():
            print(f"{name} model not found: {path}")
            return 1
    
    # Initialize pipeline
    try:
        pipeline = DetectionPipeline(args.yolo_model, args.dish_model, args.tray_model, args.device)
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return 1
    
    # Process parameters
    process_kwargs = {
        'conf_thresh': args.conf,
        'use_sliding_window': not args.no_sliding_window,
        'enhance_mode': args.enhance
    }
    
    # Process files
    successful = failed = 0
    for input_file in input_files:
        try:
            output_file = get_output_path(input_file)
            
            if Path(input_file).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                pipeline.process_video(input_file, output_file, args.skip_frames, **process_kwargs)
            else:
                pipeline.process_image(input_file, output_file, **process_kwargs)
            
            successful += 1
        except Exception as e:
            print(f"Failed to process {input_file}: {e}")
            failed += 1
    
    print(f"\nComplete: {successful} successful, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())