# 🚗 Car Color Detection and People Counting System

A machine learning-based traffic monitoring system that detects cars, identifies their colors, and counts people at traffic signals using YOLOv8 and Streamlit.

## 🎯 Features

- **Car Detection**: Automatically detects cars in traffic images
- **Color Classification**: Identifies if a car is blue or another color
- **People Counting**: Counts the number of people at the traffic signal
- **Smart Color Coding**:
  - 🟥 **Red rectangles** for blue cars
  - 🟦 **Blue rectangles** for other color cars
  - 🟢 **Green rectangles** for people
- **Interactive GUI**: Beautiful Streamlit interface with image preview
- **Real-time Statistics**: Displays counts of detected objects

## 🚀 How It Works (Fresher's Guide)

### The Thinking Process:
1. **Problem**: Need to detect cars and people, identify car colors
2. **Solution**: Use a pre-trained AI model (YOLO) that already knows how to find objects
3. **Color Detection**: Once we find a car, analyze its pixels to determine the dominant color
4. **Display**: Show everything in a nice GUI with Streamlit

### Technical Approach:
1. **YOLOv8**: Detects cars and people in images
2. **K-Means Clustering**: Finds the dominant color in each car region
3. **HSV Color Space**: Better color classification than RGB
4. **Streamlit**: Creates an interactive web interface

## 📋 Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## 🛠️ Installation

1. **Clone or download this project**:
```bash
cd c:\funproejct\internship
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Note: The first time you run the app, it will automatically download the YOLOv8 model (~6MB).

## 🎮 Usage

1. **Run the Streamlit app**:
```bash
streamlit run app.py
```

2. **Open your browser** to the URL shown (usually http://localhost:8501)

3. **Upload an image**:
   - Click "Browse files" button
   - Select a traffic image (JPG, JPEG, or PNG)

4. **View results**:
   - Original image on the left
   - Detected objects on the right
   - Statistics at the bottom

## 📁 Project Structure

```
c:\funproejct\internship\
│
├── app.py                 # Main Streamlit application
├── color_detector.py      # Color detection module
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🧠 How Each Component Works

### app.py
- Creates the Streamlit GUI
- Loads YOLOv8 model
- Processes uploaded images
- Draws rectangles and labels
- Displays statistics

### color_detector.py
- `get_dominant_color()`: Uses k-means clustering to find the most common color in a car region
- `is_blue_color()`: Checks if a color is blue using HSV color space
- `classify_car_color()`: Combines above functions to classify car color

## 🎨 Color Detection Explained

### Why HSV instead of RGB?
- RGB treats colors as (Red, Green, Blue) components
- HSV uses (Hue, Saturation, Value)
- Hue represents the actual color, making it easier to identify "blue"
- More robust to lighting changes

### Blue Detection Range:
- Hue: 90-130 (blue range in HSV)
- Saturation: >50 (not too pale)
- Value: >50 (not too dark)

## 📊 Example Output

When you upload a traffic image:
- **Total Cars**: Shows count of all detected cars
- **Blue Cars**: Count of cars classified as blue (marked with red rectangles)
- **Other Cars**: Count of non-blue cars (marked with blue rectangles)
- **People**: Count of detected people (marked with green rectangles)

## 🎯 Use Cases

- Traffic monitoring systems
- Traffic signal optimization
- Vehicle color analysis
- Pedestrian counting
- Research and education

## ⚙️ Customization

### Adjust Detection Confidence
Use the slider in the sidebar to change the detection threshold (default: 0.25)

### Modify Color Ranges
Edit the HSV ranges in `color_detector.py` to detect different colors

### Add More Vehicle Types
The model already detects: car, truck, bus, motorcycle

## 🤝 Credits

- **YOLOv8**: Ultralytics
- **Framework**: Streamlit
- **Computer Vision**: OpenCV
- **Built by**: Aryaani 

## 📝 License

Free to use for educational and personal projects by Aryaani 

---

**Happy Detecting! 🚗👥**
