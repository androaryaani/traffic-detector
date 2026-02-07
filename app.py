"""
Car Color Detection and People Counting System
Built with Streamlit and YOLOv8
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import color_detector

# Page configuration
st.set_page_config(
    page_title="Car Color Detection System",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1f77b4;
        font-size: 3rem !important;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-number {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("# 🚗 Car Color Detection & People Counter")
st.markdown("### Traffic Signal Monitoring System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider(
        "Detection Confidence", 
        0.1, 1.0, 0.4, 0.05,
        help="Higher value = fewer but more accurate detections. Lower value = more detections including false positives."
    )
    
    # Help tips
    st.markdown("#### 💡 Usage Tips:")
    st.info("""
    **For Traffic Images:**
    - Best range: **0.4 - 0.5**
    
    **If cars are missing:**
    - Try **0.3**
    
    **If extra/fake detections:**
    - Try **0.5**
    """)

# Load YOLO model
@st.cache_resource
def load_model():
    """Load YOLOv8 model - cached to avoid reloading"""
    with st.spinner("🔄 Loading AI Model... (First time may take a minute)"):
        model = YOLO('yolov8n.pt')  # Using nano model for speed
    return model

model = load_model()

# File uploader
st.subheader("📤 Upload Traffic Image")
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['jpg', 'jpeg', 'png'],
    help="Upload a traffic image to detect cars and people"
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Create columns for original and processed images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)
    
    # Process image
    with st.spinner("🔍 Detecting objects..."):
        # Run YOLO detection
        results = model(img_bgr, conf=confidence)[0]
        
        # Initialize counters
        blue_cars = 0
        other_cars = 0
        people = 0
        
        # Process each detection
        for box in results.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = results.names[cls]
            
            # Process cars
            if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                # Extract car region
                car_region = img_bgr[y1:y2, x1:x2]
                
                # Skip if region is too small
                if car_region.size == 0:
                    continue
                
                # Detect color
                car_color = color_detector.classify_car_color(car_region)
                
                # Draw rectangle based on color
                if car_color == "blue":
                    color = (0, 0, 255)  # Red rectangle for blue cars
                    blue_cars += 1
                    label = f"Blue Car {conf:.2f}"
                else:
                    color = (255, 0, 0)  # Blue rectangle for other cars
                    other_cars += 1
                    label = f"Car {conf:.2f}"
                
                # Draw rectangle and label (very subtle)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 1)
                
                # Add small label background
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                cv2.rectangle(img_bgr, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
                cv2.putText(img_bgr, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Process people
            elif class_name == 'person':
                people += 1
                color = (0, 255, 0)  # Green rectangle for people
                label = f"Person {conf:.2f}"
                
                # Draw rectangle and label (very subtle)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 1)
                
                # Add small label background
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                cv2.rectangle(img_bgr, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
                cv2.putText(img_bgr, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Convert back to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    with col2:
        st.subheader("🎯 Detected Objects")
        st.image(img_rgb, use_container_width=True)
    
    # Display statistics
    st.markdown("---")
    st.subheader("📊 Detection Statistics")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown(f"""
        <div class="metric-card" style="background: #2c3e50;">
            <div class="metric-label">Total Cars</div>
            <div class="metric-number">{blue_cars + other_cars}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown(f"""
        <div class="metric-card" style="background: #2c3e50;">
            <div class="metric-label">Blue Cars 🟥</div>
            <div class="metric-number">{blue_cars}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown(f"""
        <div class="metric-card" style="background: #2c3e50;">
            <div class="metric-label">Other Cars 🟦</div>
            <div class="metric-number">{other_cars}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col4:
        st.markdown(f"""
        <div class="metric-card" style="background: #2c3e50;">
            <div class="metric-label">People 🟢</div>
            <div class="metric-number">{people}</div>
        </div>
        """, unsafe_allow_html=True)

else:
    # Show placeholder when no image is uploaded
    st.info("👆 Please upload a traffic image to get started!")
    
    # Show example instructions
    st.markdown("### 🎯 How it works:")
    st.markdown("""
    1. **Upload** a traffic signal image using the uploader above
    2. **AI Detection** - YOLOv8 will detect all cars and people
    3. **Color Analysis** - The system analyzes each car's dominant color
    4. **Smart Marking**:
       - Blue cars get **RED** rectangles 🟥
       - Other color cars get **BLUE** rectangles 🟦
       - People get **GREEN** rectangles 🟢
    5. **Statistics** - View counts of all detected objects
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Built with ❤️ using Aryaani</p>
</div>
""", unsafe_allow_html=True)
