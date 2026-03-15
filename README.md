# AI Image Tagger & Face Profiler
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![YOLO11s](https://img.shields.io/badge/YOLO-11s-FF9900)
![Deployed on Vercel](https://img.shields.io/badge/Deployed-Vercel-black)

A full-stack computer vision web application that automatically scans, tags, and organizes images using machine learning. It features object detection, 128D face recognition, and a searchable web gallery.

[**Live Here**](https://your-vercel-app-link-goes-here.vercel.app)

---

## Key Features
* **Automated Object Tagging:** Uses **YOLO11s** to detect objects within images and assigns searchable tags with confidence scores.
* **Facial Recognition & Profiling:** Extracts 128D facial embeddings and clusters images of the same person across the gallery using Euclidean distance math (Threshold < 0.6).
* **Smart Deduplication:** Calculates cryptographic hashes for every uploaded image to prevent duplicate database entries and save storage.
* **FastAPI Backend:** A lightweight, high-performance API that serves a dynamic Jinja2 frontend template.
* **Serverless Deployment:** Optimized for Vercel's edge network, separating the heavy ML inference pipeline (local) from the lightweight web serving pipeline (cloud).

## Tech Stack
* **Backend:** Python, FastAPI, SQLite
* **Machine Learning:** Ultralytics (YOLO11s), OpenCV, NumPy
* **Frontend:** HTML, CSS, JavaScript, Jinja2 Templates
* **Deployment:** Vercel, Git

---

## How the ML Pipeline Works
1. **Scanning:** A local Python script (`scanner.py`) reads images from a directory and generates a unique hash.
2. **Detection:** YOLO11s scans the image, returning bounding boxes, labels, and confidence thresholds.
3. **Embedding:** Faces are isolated, aligned, and converted into 128D mathematical vectors. 
4. **Matching:** The system calculates the distance between the new face vector and existing profiles in the SQLite database. If a match is found, they are linked; if not, a new profile is created.
5. **Serving:** The FastAPI application queries the SQLite database to render the web gallery, allowing users to filter by specific tags or face profiles.
