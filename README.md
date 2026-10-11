# Simplified ADAS with Computer Vision

Educational prototype of an Advanced Driver Assistance System (ADAS) built with computer-vision techniques.

The project processes road video and combines several detection tasks in one pipeline.

## Features

- Pedestrian detection
- Lane detection
- Obstacle detection
- Vehicle detection
- Traffic-light analysis

## Pipeline

```text
video frame
   ↓
preprocessing
   ↓
lane / vehicle / pedestrian / obstacle analysis
   ↓
traffic-light analysis
   ↓
overlay and visualization
```

## Installation

Using pip:

```bash
pip install -r requirements.txt
```

Or Conda:

```bash
conda env create -f environment.yml
conda activate adas_yolo
```

## Run

```bash
python main.py --input video1_720.mp4 --show
```

Replace the input path with another compatible video when needed.

## Scope

This project is a simplified academic prototype. It is intended to demonstrate how multiple computer-vision components can be combined in a road-scene pipeline.

It is not a production automotive safety system and must not be used for real driving decisions.
