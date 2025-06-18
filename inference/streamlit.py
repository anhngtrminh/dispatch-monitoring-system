import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import os
import uuid
from datetime import datetime
import base64
from io import BytesIO
# import sys
import subprocess
from pathlib import Path

from inference import DetectionClassificationPipeline

# Configuration
YOLO_MODEL_PATH = "models/detection.pt"
DISH_MODEL_PATH = "models/dish_classifier.pt"
TRAY_MODEL_PATH = "models/tray_classifier.pt"
FEEDBACK_DIR = "feedback"
CROPS_DIR = "feedback/crops"

# Create necessary directories
os.makedirs(FEEDBACK_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)

# Streamlit page configuration
st.set_page_config(
    page_title="Object Detection & Classification",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_pipeline():
    """Load the detection pipeline (cached to avoid reloading)"""
    try:
        pipeline = DetectionClassificationPipeline(
            yolo_model_path=YOLO_MODEL_PATH,
            dish_model_path=DISH_MODEL_PATH,
            tray_model_path=TRAY_MODEL_PATH,
            device='auto'
        )
        return pipeline
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def save_crop_image(image, bbox, object_id):
    """Save cropped image for feedback"""
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    crop_path = os.path.join(CROPS_DIR, f"{object_id}.jpg")
    cv2.imwrite(crop_path, crop)
    return crop_path

def save_feedback(feedback_data):
    """Save user feedback to JSON file"""
    feedback_file = os.path.join(FEEDBACK_DIR, f"{feedback_data['object_id']}.json")
    with open(feedback_file, 'w') as f:
        json.dump(feedback_data, f, indent=2)

def image_to_base64(image):
    """Convert image to base64 for display"""
    if isinstance(image, np.ndarray):
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
    else:
        pil_image = image
    
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def draw_detection_boxes(image, detections):
    """Draw bounding boxes on image"""
    annotated_image = image.copy()
    
    # Colors for different types and statuses
    colors = {
        'dish': (0, 255, 0),    # Green
        'tray': (255, 0, 0),    # Blue
    }
    
    status_colors = {
        'empty': (128, 128, 128),      # Gray
        'kakigori': (0, 255, 255),     # Yellow
        'not_empty': (0, 0, 255)       # Red
    }
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        obj_type = detection['type']
        status = detection['status']
        confidence = detection['confidence']
        
        # Choose color based on object type
        box_color = colors.get(obj_type, (255, 255, 255))
        status_color = status_colors.get(status, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color, 2)
        
        # Prepare text
        text = f"{obj_type}: {status} ({confidence:.2f})"
        
        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(annotated_image, 
                     (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), 
                     status_color, -1)
        
        # Draw text
        cv2.putText(annotated_image, text, (x1, y1 - 5), 
                   font, font_scale, (255, 255, 255), thickness)
    
    return annotated_image

def main():
    st.title("🍽️ Object Detection & Classification System")
    st.markdown("Upload an image to detect dishes and trays, then provide feedback to improve the model.")
    
    # Load pipeline
    pipeline = load_pipeline()
    if pipeline is None:
        st.error("Failed to load detection pipeline. Please check your model files.")
        return
    
    # Sidebar configuration
    st.sidebar.header("Detection Settings")
    
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.3, 
        step=0.05,
        help="Lower values detect more objects but may include false positives"
    )
    
    use_sliding_window = st.sidebar.checkbox(
        "Use Sliding Window", 
        value=True,
        help="Better for detecting small objects in large images"
    )
    
    enhancement_mode = st.sidebar.selectbox(
        "Image Enhancement",
        options=['auto', 'always', 'never', 'upscale_only'],
        index=0,
        help="Auto: enhance only when needed, Always: always enhance, Never: no enhancement"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
        help="Upload an image containing dishes and/or trays"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # col1, col2 = st.columns(2)
        
        # with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
        with st.spinner("Processing image..."):
            # Process image
            detections = pipeline.process_frame(
                image_cv,
                conf_threshold=conf_threshold,
                use_sliding_window=use_sliding_window,
                enhancement_mode=enhancement_mode
            )
            
            # Draw annotations
            annotated_image = draw_detection_boxes(image_cv, detections)
        
        # with col2:
        st.subheader("Detection Results")
        st.image(annotated_image, channels="BGR", use_column_width=True)
        
        # Display detection results
        st.subheader("📊 Detection Summary")
        
        if detections:
            # Summary statistics
            total_objects = len(detections)
            dishes = [d for d in detections if d['type'] == 'dish']
            trays = [d for d in detections if d['type'] == 'tray']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Objects", total_objects)
            with col2:
                st.metric("Dishes", len(dishes))
            with col3:
                st.metric("Trays", len(trays))
            
            # Detailed results and feedback
            st.subheader("🔧 Provide Feedback")
            st.markdown("Review the detections below and correct any mistakes to help improve the model.")
            
            # Store feedback in session state
            if 'feedback_data' not in st.session_state:
                st.session_state.feedback_data = {}
            
            for i, detection in enumerate(detections):
                with st.expander(f"Object {i+1}: {detection['type']} - {detection['status']} (confidence: {detection['confidence']:.2f})"):
                    
                    # Show crop
                    x1, y1, x2, y2 = detection['bbox']
                    crop = image_cv[y1:y2, x1:x2]
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(crop, channels="BGR", caption="Detected Object", width=200)
                    
                    with col2:
                        st.write(f"**Bounding Box:** ({x1}, {y1}, {x2}, {y2})")
                        st.write(f"**Object Type:** {detection['type']}")
                        st.write(f"**Predicted Status:** {detection['status']}")
                        st.write(f"**Confidence:** {detection['confidence']:.3f}")
                        
                        # Feedback form
                        object_key = f"object_{i}"
                        
                        # Object type correction
                        correct_type = st.selectbox(
                            "Correct Object Type:",
                            options=['dish', 'tray', 'unknown'],
                            index=0 if detection['type'] == 'dish' else 1,
                            key=f"type_{object_key}"
                        )
                        
                        # Status correction
                        correct_status = st.selectbox(
                            "Correct Status:",
                            options=['empty', 'kakigori', 'not_empty'],
                            index=['empty', 'kakigori', 'not_empty'].index(detection['status']),
                            key=f"status_{object_key}"
                        )
                        
                        # Check if user made corrections
                        has_corrections = (
                            correct_type != detection['type'] or 
                            correct_status != detection['status']
                        )
                        
                        if has_corrections:
                            st.warning("⚠️ You have made corrections to this detection.")
                            
                            if st.button(f"Submit Feedback for Object {i+1}", key=f"submit_{object_key}"):
                                # Generate unique object ID
                                object_id = str(uuid.uuid4())
                                
                                # Save crop image
                                crop_path = save_crop_image(image_cv, detection['bbox'], object_id)
                                
                                # Prepare feedback data
                                feedback_data = {
                                    'object_id': object_id,
                                    'timestamp': datetime.now().isoformat(),
                                    'original_image': uploaded_file.name,
                                    'bbox': detection['bbox'],
                                    'original_type': detection['type'],
                                    'original_status': detection['status'],
                                    'original_confidence': detection['confidence'],
                                    'corrected_type': correct_type,
                                    'corrected_state': correct_status,
                                    'crop_path': crop_path
                                }
                                
                                # Save feedback
                                save_feedback(feedback_data)
                                
                                st.success(f"✅ Feedback submitted for Object {i+1}!")
                                st.balloons()
                        else:
                            st.info("✅ No corrections needed for this detection.")
            
            # Show feedback statistics
            if os.path.exists(FEEDBACK_DIR):
                feedback_files = [f for f in os.listdir(FEEDBACK_DIR) if f.endswith('.json')]
                if feedback_files:
                    st.subheader("📈 Feedback Statistics")
                    st.info(f"Total feedback submissions: {len(feedback_files)}")
                    
                    if st.button("🔄 Retrain Models"):
                        st.warning("Model retraining! Need to wait until it completes.")
                        # Here you would call your retraining script
                        subprocess.run(["python", "inference/retrain.py"], check=True)
        
        else:
            st.warning("No objects detected. Try adjusting the confidence threshold or enhancement settings.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**How to use:**\n"
        "1. Upload an image containing dishes and/or trays\n"
        "2. Adjust detection settings if needed\n"
        "3. Review the detection results\n"
        "4. Provide feedback on any incorrect detections\n"
        "5. Submit feedback to help improve the model"
    )

if __name__ == "__main__":
    main()