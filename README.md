# pyt-your-hands-up

Pushing particles around using MediaPipe hand tracking.

## Contents

### [handparticles.py](./src/handparticles.py)

A script that allows you to visually interact with a particle system using your right index finger.

### [handtracking.py](./src/handtracking.py)

Contains the necessary hand detection functionality based on the [MediaPipe](https://github.com/google-ai-edge/mediapipe) Python bindings.

### [particlesystem.py](./src/particlesystem.py)

Contains a very simple 2D particle system.

## Installation

1. Setup a Python virtual environment/ conda environment (tested with Python 3.12)

2. Activate the environment and run
    ```bash
    pip install -r requirements.txt
    ```

3. Download the MediaPipe Hand Landmarker model from [the google dev page](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index#models).

## Usage

The main entry point is in [handparticles.py](./src/handparticles.py).
To List all available options, run:
```bash
python src/handparticles.py --help
```

A usage example can be found in [handparticles.sh](./handparticles.sh), which invokes `handparticles.py` with some parameters that worked for me.

Change the `model_path` parameter to point at the downloaded MediaPipe model (see [Installation](#installation)), then run:

```bash
bash handparticles.sh
```
