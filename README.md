# Simple-SLAM: Monocular Room Mapper

A Python-based Visual SLAM (Simultaneous Localization and Mapping) implementation that performs 3D sparse reconstruction and camera trajectory tracking from a single smartphone video.

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-orange.svg)

## 📌 Project Overview
This project implements a monocular SLAM pipeline to map a room's geometry in 3D. It uses **ORB (Oriented FAST and Rotated BRIEF)** for feature detection, **Essential Matrix** estimation for initialization, and **SolvePnP** for real-time camera tracking.

### Key Features
* **ORB Feature Tracking:** Detects and matches up to 5,000 unique keypoints per frame.
* **Camera Pose Estimation:** Uses RANSAC-based PnP to determine the camera's location in 3D space.
* **Continuous Triangulation:** Grows the 3D point cloud dynamically as the camera moves through the room.
* **Stability Filtering:** Implements a movement threshold to reject "teleportation" errors and numerical instability.
* **Live 3D Visualization:** Real-time plotting of the camera path and map points using Matplotlib.

## 🛠️ Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV (`cv2`)
* **Numerical Operations:** NumPy
* **Visualization:** Matplotlib (3D Projection)

## 🚀 Evolution of the Project
During development, we tackled several critical SLAM challenges:
1.  **Numerical Explosion:** Fixed a feedback loop where rotation matrices were multiplying exponentially ($1e^{97}$ errors).
2.  **Coordinate Transformation:** Implemented the correct transformation from "Camera Space" to "World Space" using $R^T$ and $-R^T \cdot t$.
3.  **Outlier Rejection:** Added a Euclidean distance filter to prevent the camera trajectory from turning into "blue spaghetti."

## 📂 Project Structure
.
├── mapping.py          # Main SLAM logic and RoomMapper class
├── room_walk.mp4       # Input video dataset
└── README.md           # Project documentation

⚙️ How It Works

Initialization: The first 20 frames are used to establish a baseline and triangulate the initial 3D point cloud.

Tracking: For every new frame, the model matches 2D features to existing 3D map points.

Mapping: Every 10 frames, new features are triangulated and added to the global map to ensure the "room" keeps growing.

Filtering: If the calculated movement is $> 0.5$ units between frames, the position is rejected as a noise artifact.

📊 Results

The final output generates a Sparse 3D Point Cloud (representing walls and objects) and a Continuous Trajectory (representing the user's walk through the room).
