# 🏗️ Construction Site Safety App

### AI-Powered Construction Safety Monitoring and PPE Detection

An AI-powered construction site safety monitoring application that uses computer vision to identify workers, detect personal protective equipment (PPE), and highlight potential safety violations in construction-site images.

The system combines a fine-tuned YOLOv8 object detection model, a Python-based safety-rule engine, a FastAPI backend, and a React frontend to help identify missing safety equipment and restricted-zone violations.

> **Project goal:** Improve construction-site safety awareness through automated visual inspection and risk identification.

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Our Solution](#-our-solution)
- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [Technology Stack](#-technology-stack)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Running the Application](#-running-the-application)
- [API Documentation](#-api-documentation)
- [Model Training](#-model-training)
- [Limitations](#-limitations)
- [Future Scope](#-future-scope)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚨 Problem Statement

Construction sites involve multiple hazards, including falling objects, unsafe working areas, and workers operating without appropriate personal protective equipment.

Traditional safety inspections can be time-consuming and depend heavily on manual observation. Missing helmets, safety vests, or unauthorized entry into restricted areas may go unnoticed during busy operations.

A computer vision-based monitoring system can help safety teams identify potential violations more efficiently and review risky situations before they lead to serious incidents.

## 💡 Our Solution

The Construction Site Safety App uses AI-based image analysis to detect workers and relevant safety equipment in uploaded construction-site images.

The system evaluates detected objects against configurable safety rules and produces an annotated image, a violation summary, and risk classifications.

The application is designed to help construction supervisors and safety officers:

- Identify workers who may be missing required PPE.
- Detect potential restricted-zone violations.
- Review safety risks through visual annotations.
- Understand violations through structured results.
- Improve the efficiency of image-based safety inspections.

---

## ✨ Key Features

### 🦺 1. PPE Detection

Uses a fine-tuned YOLOv8 object detection model to identify supported classes of safety equipment.

Depending on the trained model's available classes, detections may include:

- Safety helmets
- High-visibility safety vests
- Gloves
- Safety boots
- Goggles
- Workers

### ⚠️ 2. Safety Violation Detection

Applies a rule-based safety engine to associate detected workers with PPE detections and identify potential violations.

Examples include:

- Missing helmet
- Missing safety vest
- Entry into a configured restricted zone

The exact violations depend on the model's supported classes and the rules implemented in the application.

### 🔴 3. Risk Classification

Assigns risk categories to detected workers based on the configured safety rules.

- `SAFE` — No configured violation was detected.
- `HIGH` — A high-priority safety violation was identified.
- `CRITICAL` — A critical condition was identified according to the configured rules.

These classifications are automated screening results, not certified safety assessments.

### 🖼️ 4. Annotated Image Output

Generates an annotated image containing bounding boxes and labels to make potential hazards easier to inspect.

### 🚧 5. Restricted-Zone Monitoring

Allows users to define restricted areas using polygon coordinates associated with the uploaded image.

The safety engine checks detected worker positions against the configured zones.

### 🌐 6. Web-Based User Interface

Provides a React-based interface for uploading images, initiating analysis, and reviewing detection results.

### 📊 7. Structured Analysis Results

Returns structured JSON containing image information, raw detections, worker-level violations, risk classifications, summary statistics, and an annotated-image reference.

---

## 🔄 How It Works

```text
       Construction-Site Image
                 |
                 v
          Upload via Web UI
                 |
                 v
          FastAPI Backend
                 |
                 v
       YOLOv8 Object Detection
                 |
                 v
     Detect Workers and PPE Items
                 |
                 v
       Safety Rule Evaluation
                 |
          +------+------+
          |             |
          v             v
       PPE Check    Zone Check
          |             |
          +------+------+
                 |
                 v
       Identify Violations
                 |
                 v
         Assign Risk Levels
                 |
                 v
       Annotate the Image
                 |
                 v
      JSON Results and Image
                 |
                 v
          React Dashboard
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend development and safety logic |
| YOLOv8 | Object detection |
| Ultralytics | Model inference and training |
| OpenCV | Image processing and annotation |
| FastAPI | REST API development |
| Uvicorn | ASGI server |
| React | Frontend interface |
| Vite | Frontend development and build tooling |
| PyTorch | Deep learning model execution |
| Kaggle | GPU-based model training workflow |

---

## 🏗️ Project Architecture

The application separates image processing, safety-rule evaluation, API handling, and the user interface.

```text
+-----------------------------------+
|          React Frontend           |
|                                   |
| Image Upload                      |
| Analysis Controls                 |
| Violation Summary                 |
| Annotated Image Display           |
+----------------+------------------+
                 |
                 | HTTP Requests
                 v
+-----------------------------------+
|           FastAPI Backend         |
|                                   |
| Image Upload Endpoint             |
| Request Validation                |
| Detection Orchestration           |
| JSON Response                     |
+----------------+------------------+
                 |
                 v
+-----------------------------------+
|       YOLOv8 Detection Layer      |
|                                   |
| Worker Detection                  |
| PPE Detection                     |
| Confidence Scores                 |
| Bounding Boxes                    |
+----------------+------------------+
                 |
                 v
+-----------------------------------+
|         Safety Rule Engine        |
|                                   |
| Worker-PPE Association            |
| Missing PPE Checks                |
| Restricted-Zone Checks            |
| Risk Classification               |
+----------------+------------------+
                 |
                 v
+-----------------------------------+
|         Analysis Results          |
|                                   |
| Annotated Image                   |
| Violation Details                 |
| Risk Summary                      |
| Structured JSON                   |
+-----------------------------------+
```

---

## 📁 Project Structure

```text
Construction-Site-Safety-App/
│
├── backend/
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── App.css
│   └── package.json
│
├── ml/
│   ├── detector.py
│   ├── safety_rules.py
│   ├── weights/
│   │   └── best.pt
│   ├── kaggle_train.ipynb
│   └── train_yolo.py
│
├── .env.example
├── .gitignore
├── KAGGLE_TRAINING.md
├── README.md
└── ...
```

*Note: This is a simplified overview. Check the repository for the exact files and directories. The trained model weights may need to be obtained or generated separately.*

---

## 🚀 Getting Started

### Prerequisites

Install the following:

- Python 3.10 or another version supported by the project dependencies
- Node.js and npm
- Git
- A compatible environment for running the YOLOv8 model
- Trained model weights required by the detector

### 1. Clone the Repository

```bash
git clone https://github.com/Rehan-070/Construction-Site-Safety-App.git
cd Construction-Site-Safety-App
```

### 2. Set Up the Python Backend

Create a virtual environment.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r backend/requirements.txt
```

### 3. Configure the Model

The detector requires compatible trained YOLO model weights.

Check the model configuration in `ml/detector.py` and the example environment configuration in `.env.example`.

If `best.pt` is not included in the repository, obtain authorized weights or train a compatible model using the provided training resources.

Ensure that the model path configured in the application matches the location of the weights.

### 4. Start the Backend

From the repository root, run:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The backend should be available at:

`http://127.0.0.1:8000`

If enabled, the interactive API documentation is available at:

`http://127.0.0.1:8000/docs`

### 5. Set Up the Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the development URL printed by Vite, typically:

`http://localhost:5173`

If the frontend cannot connect to the backend, verify the configured API URL and backend CORS settings.

---

## 🧪 Running the Application

1. Start the backend and frontend.
2. Open the frontend in your browser.
3. Upload a construction-site image in JPEG or PNG format.
4. Optionally configure a restricted zone using polygon coordinates.
5. Select **Analyze Image**.
6. Review the detected objects, violation details, risk classifications, and annotated image.

### Example Restricted-Zone Input

The following example represents one polygon defined using image pixel coordinates:

```json
[
  [
    [100, 50],
    [500, 50],
    [500, 600],
    [100, 600]
  ]
]
```

The actual accepted format must match the API's `restricted_zones` parameter.

---

## 🔌 API Documentation

### Health Check

`GET /health`

Checks the availability of the backend and reports model-loading status.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Analyze an Image

`POST /analyze/image`

Analyzes an uploaded construction-site image for supported objects and safety violations.

**Request parameters**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file` | File upload | Yes | Construction-site image |
| `restricted_zones` | JSON string | No | Polygon coordinates defining restricted areas |

**Example response**

```json
{
  "status": "success",
  "image_info": {
    "filename": "construction_site.jpg",
    "width": 640,
    "height": 480
  },
  "raw_detections": [
    {
      "class": "helmet",
      "confidence": 0.91,
      "bbox": [120, 40, 180, 100]
    }
  ],
  "workers": [
    {
      "type": "worker",
      "bbox": [100, 50, 250, 420],
      "confidence": 0.8,
      "violations": ["MISSING_VEST"],
      "risk": "HIGH"
    }
  ],
  "summary": {
    "workers_detected": 1,
    "total_violations": 1,
    "highest_risk": "HIGH"
  },
  "annotated_image_url": "/outputs/example_annotated.jpeg"
}
```

This is an illustrative response. Actual values and field names depend on the backend implementation.

---

## 🧠 Model Training

The project includes training resources for fine-tuning YOLOv8 for construction-site PPE detection.

Training-related files include:

- `ml/kaggle_train.ipynb` — training notebook for a GPU-enabled environment.
- `ml/train_yolo.py` — local training script.
- `KAGGLE_TRAINING.md` — training instructions and related details.

### General Training Workflow

1. Prepare an appropriately annotated construction safety dataset.
2. Configure the dataset paths and class labels.
3. Select a YOLOv8 model configuration.
4. Train or fine-tune the model.
5. Evaluate detection accuracy on validation data.
6. Export or save the trained weights.
7. Configure the application to load the resulting model.
8. Test detection and safety rules on representative images.

The model's real-world reliability depends on the dataset, annotation quality, lighting, camera angle, occlusion, and evaluation results.

---

## ⚠️ Limitations

- **Image-based analysis:** The documented workflow analyzes uploaded images rather than providing continuous CCTV monitoring.
- **Model dependency:** Analysis requires compatible trained weights.
- **Detection uncertainty:** Objects can be missed or incorrectly classified.
- **PPE association:** Spatial rules may incorrectly associate equipment with workers in crowded scenes.
- **Restricted zones:** Polygon coordinates depend on the uploaded image dimensions and correct configuration.
- **Risk classification:** Risk labels reflect configured rules and are not a certified assessment of workplace safety.
- **Human oversight:** Results should be reviewed by qualified site-safety personnel.

This application is a prototype and should not be used as the sole basis for safety-critical decisions.

---

## 🔮 Future Scope

Potential improvements include:

- [ ] Real-time CCTV and video-stream analysis
- [ ] Real-time alerts for missing PPE
- [ ] Worker tracking across video frames
- [ ] Improved detection of gloves, boots, and goggles
- [ ] Detection of unsafe proximity to machinery
- [ ] Configurable safety zones and site-specific rules
- [ ] Incident history and violation analytics
- [ ] Exportable safety inspection reports
- [ ] Multi-camera monitoring
- [ ] Cloud deployment and scalable inference
- [ ] Model evaluation dashboards
- [ ] Automated testing and CI/CD integration

---

## 🤝 Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.

   ```bash
   git checkout -b feature/your-feature
   ```

3. Implement and test your changes.
4. Commit your changes.

   ```bash
   git commit -m "Add: your feature"
   ```

5. Push your branch.

   ```bash
   git push origin feature/your-feature
   ```

6. Open a pull request with a clear description of the changes.

---

## 📄 License

Check the repository for its `LICENSE` file. If no license is present, contact the repository owner before redistributing or using the project beyond the permissions granted by applicable law.

---

## 👨‍💻 Project Information

**Project:** Construction Site Safety App  
**Repository:** [Rehan-070/Construction-Site-Safety-App](https://github.com/Rehan-070/Construction-Site-Safety-App)  
**Domain:** Artificial Intelligence · Computer Vision · Construction Safety

### Our Vision

To help construction teams identify potential safety risks earlier through accessible AI-powered visual analysis and more consistent safety inspections.

**Build safer sites with intelligent monitoring.**
