# AbnormalDetectionSystem
This project is a modular application that performs real-time video capture and anomaly detection using OpenCV, PySide6, and TensorFlow.



---

## Description

- **Multi-threading**: `Thread_in` (video capture) and `Thread_out` (video processing) run concurrently, improving efficiency.  
- **Real-time Video Processing**: Utilizes OpenCV to capture from webcams or video files, displaying frames in a PySide6 GUI.  
- **ROI & Object Detection**: Offers various functions such as **Haar Cascade** detection, **Mean Shift** tracking, **Canny/Laplacian** edge detection, and ROI selection.  
- **Labeling & Training**: Provides frame-by-frame labeling (0/1) and a simple TensorFlow-based model for anomaly detection experiments.  
- **Keyboard Automation**: Uses `pyautogui` to automate **Alt+Tab** or other keyboard events when certain conditions (e.g., edge intensity or ROI triggers) are met.

---
## Repository Structure
```bash
AbnormalDetectionSystem/
├ code
  ├── config.py          # Global configurations and shared variables
  ├── capture.py         # Video capture module (Thread_in class)
  ├── processing.py      # Video processing module (Thread_out class and helper functions)
  ├── ui.py              # GUI implementation with PySide6
  ├── main.py            # Application entry point
├── requirements.txt   # List of required Python packages
└── README.md          # This file

```

🚀 Usage
```bash
pip install -r requirements.txt
python main.py
```
