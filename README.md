# Real-Time Autonomous Vehicle Perception Pipeline

This project is a comprehensive perception system designed to interpret complex driving environments for autonomous vehicles. It processes sensor data to detect, track, and predict the behavior of objects in real time, forming a foundational component for autonomous navigation and decision-making.

![Demo GIF](https://i.imgur.com/your-demo-gif.gif)
*(**Suggestion:** Record a short GIF of your Gradio app in action, upload it to a site like [imgur.com](https://imgur.com), and replace the link above to showcase your project visually.)*

---

## **Core Features**

-   **Real-Time Object Detection:** Utilizes a pre-trained **YOLOv8** model to identify and classify objects such as cars, trucks, and pedestrians from camera footage with high efficiency.
-   **Multi-Object Tracking:** Implements the **DeepSORT** algorithm to assign and maintain a consistent ID for each detected object as it moves across frames, crucial for understanding object behavior over time.
-   **Trajectory Prediction:** Calculates and visualizes both the recent historical path (tail) and the predicted future path of tracked objects, enabling anticipatory vehicle behavior.
-   **Sensor Fusion:** Combines 2D camera data with 3D **LiDAR** point cloud data by performing coordinate transformations. This enables the system to project LiDAR points onto the image plane to accurately estimate the real-world distance to each detected object.
-   **Quantitative Evaluation:** Includes scripts to measure the object detector's performance using the industry-standard **Mean Average Precision (mAP)** metric, demonstrating a rigorous, data-driven approach to model validation.
-   **Interactive Web Demo:** A user-friendly interface built with **Gradio** allows for easy demonstration of the tracking and prediction pipeline on any uploaded video file.

---

## **Tech Stack**

-   **Core ML & Computer Vision:** Python, PyTorch, Ultralytics YOLOv8, OpenCV
-   **Tracking & 3D Processing:** DeepSORT, Open3D, NumPy
-   **Web Application & Deployment:** Gradio, MoviePy
-   **Dataset:** KITTI Vision Benchmark Suite (Raw Data)

---

## **Setup and Installation**

Follow these steps to set up the project environment locally.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ayushpathak477/av-perception.git](https://github.com/ayushpathak477/av-perception.git)
    cd av-perception
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # On Windows
    python -m venv venv
    .\venv\Scripts\activate

    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies from the requirements file:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Sample Data:**
    * **KITTI Raw Data:** Download the data for `2011_09_26_drive_0013_sync` and place the unzipped folder in the `data/` directory.
    * **KITTI Calibration Files:** Download the corresponding calibration files (`2011_09_26_calib.zip`) and place the extracted `.txt` files into a `data/2011_09_26/` folder.
    * **Sample Video:** Download a sample driving video (e.g., from Pexels.com) and save it as `data/test_video.mp4` for use with the Gradio app.

---

## **Usage**

The project has three main entry points for different functionalities.

### **1. Interactive Web Demo**

This is the easiest way to see the object tracking and trajectory prediction in action.
```bash
python app.py