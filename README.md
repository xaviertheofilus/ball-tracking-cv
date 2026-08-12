# Ball Tracking with YOLO11

Small end-to-end computer vision project for detecting and tracking a ball in sports video. The notebook downloads a Roboflow dataset, fine-tunes YOLO11n, evaluates the model, runs ByteTrack, and exports an annotated video with frame-level trajectory data.

## Technical overview

- Model: YOLO11n
- Classes: `ball` and `player`
- Tracking: ByteTrack
- Training device: CPU
- Input size: 640 pixels
- Training: up to 50 epochs with early stopping
- Missing detections: explicitly recorded as `missing`

The tracked ball position is the center of the highest-confidence ball bounding box in each frame.

## Project files

```text
ball_tracking.ipynb       Training, evaluation, and video tracking
assignment_response.md    Written assignment response
movel_ai_assignment.md    Assignment specification
test2.mp4                 Test video
dataset/                  Downloaded Roboflow dataset
runs/                     Training and evaluation results
outputs/                  Annotated video and trajectory CSV
```

## Requirements

- Python 3.10 or newer
- Jupyter Notebook
- Internet connection for the first dataset download
- Roboflow API key with access to the dataset

## How to run

Open PowerShell in the project directory:

```powershell
cd D:\TRAINING\ball-tracker-cv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install jupyter
jupyter notebook ball_tracking.ipynb
```

Set the Roboflow key in `.env.example`:

```env
ROBOFLOW_API_KEY=your_api_key
```

Do not commit a real API key to a public repository.

Inside the notebook:

1. Update `PROJECT_ROOT` if the project is stored in another directory.
2. Run all cells from top to bottom.
3. Wait for dataset download, training, and evaluation to finish.
4. Change `VIDEO_PATH` in the final cell to test another video.
5. Run the final cell to generate the tracked output.

The notebook installs `ultralytics` and `roboflow` automatically in its active Python environment.

## Outputs

Training artifacts and model weights:

```text
runs/ball_detector_cpu_fast/
runs/ball_detector_cpu_fast/weights/best.pt
```

Evaluation artifacts:

```text
runs/evaluation_cpu_fast/
```

Tracked video and trajectory:

```text
outputs/<video_name>_tracking/<video_name>.avi
outputs/<video_name>_tracking/trajectory.csv
```

The trajectory CSV contains `frame_number`, `x_position`, `y_position`, `confidence`, and `status`.
