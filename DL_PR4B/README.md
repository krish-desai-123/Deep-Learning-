<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=26&duration=3000&pause=800&color=00E5FF&center=true&vCenter=true&width=760&lines=Project+B+%E2%80%94+Teachable+Machine;Train+visually.+Deploy+in+Python.;Image+%2B+Pose+%E2%86%92+One+Live+Dashboard" alt="typing header"/>

![Project](https://img.shields.io/badge/Project-B%20%E2%80%94%20Teachable%20Machine-0d1117?style=for-the-badge&logo=google&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-0d1117?style=for-the-badge&logo=python&logoColor=3776AB)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-0d1117?style=for-the-badge&logo=tensorflow&logoColor=FF6F00)
![OpenCV](https://img.shields.io/badge/OpenCV-Webcam-0d1117?style=for-the-badge&logo=opencv&logoColor=5C3EE8)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-0d1117?style=for-the-badge&logo=jupyter&logoColor=F37626)

</div>

---

## 📌 Project Declaration
**Deep Learning · PR 4 — I chose _Project B — Teachable Machine_** (train models visually in the browser, export them, deploy and extend them in Python).
Two models were trained and deployed: an **Image** model (vehicles) and a **Pose** model (yoga). An audio model was not trained for this submission.

## 🗂️ Repository Layout
```
DL_PR4B/
├── image/
│   ├── image_classifier.ipynb      # B1 – vehicle classifier + Vehicle Logger
│   ├── models/                     # keras_model.h5, labels.txt
│   ├── output/  screenshots/
├── pose/
│   ├── pose_classifier.ipynb       # B3 – PoseNet + yoga pose classifier + Yoga Coach
│   ├── models/                     # model.json, weights.bin, metadata.json, posenet/
│   ├── reference/  output/  screenshots/
├── CONCEPTS.md                     # Transfer-learning explanation
└── requirements.txt
```

## ⚙️ Setup
```bash
git clone https://github.com/krish-desai-123/Deep-Learning-.git
cd Deep-Learning-/DL_PR4B
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```
Open each notebook **from its own folder** (`image/`, `pose/`, `combined/`) and run all cells. A webcam window opens — press **q** to quit.
Note: the notebooks use `tf_keras` (legacy Keras 2) because Teachable Machine `.h5` files cannot be loaded by Keras 3. Keep `tf_keras` on the same minor version as TensorFlow (e.g. `pip install "tf_keras==2.20.*"` for TensorFlow 2.20), then restart the kernel.

## 🤖 Trained Models

<details open>
<summary><b>1 · Image model — Vehicle classifier (20 classes)</b></summary>

| Field | Detail |
|---|---|
| Dataset | [Kaggle – vehicles image dataset](https://www.kaggle.com/datasets/mmohaiminulislam/vehicles-image-dataset) |
| Classes | Airplane, Ambulance, Bicycle, Boat, Bus, Car, Fire Truck, Helicopter, Hovercraft, Jet Ski, Kayak, Motorcycle, Rickshaw, Scooter, Segway, Skateboard, Tractor, Truck, Unicycle, Van |
| Samples per class | `TODO – add from your Teachable Machine panel` |
| Training settings | `TODO – epochs / batch size / learning rate shown in Advanced` |
| Final accuracy | `TODO` |
| Export | TensorFlow → Keras (`keras_model.h5` + `labels.txt`) |
| Model | MobileNetV2-style extractor + `Dense(100) → Dense(20)`; 540,308 parameters (130,100 in the head) |

Deployment: centre-crop → 224×224 → `x/127.5 − 1` → predict → class + confidence bars.
Extension: **Vehicle Logger** – appends `timestamp, class, confidence` to a CSV every 5 s.

![Image training panel](Assets/image_training_panel.png)
</details>

<details open>
<summary><b>2 · Pose model — Yoga poses (5 classes)</b></summary>

| Field | Detail |
|---|---|
| Dataset | [Kaggle – yoga pose classification](https://www.kaggle.com/datasets/ujjwalchowdhury/yoga-pose-classification) |
| Classes | Downdog, Goddness, Plank, Tree, Warrior2 |
| Samples per pose | `TODO` |
| Training settings | `TODO` |
| Final accuracy | `TODO` |
| Export | TensorFlow.js (`model.json`, `weights.bin`, `metadata.json`) |
| Model | PoseNet MobileNetV1 ×0.75 (stride 16, 257 px) → 14,739 features → `Dense(100) → Dropout(0.5) → Dense(5)`; 1,474,500 parameters |

Deployment: PoseNet features + keypoints → classifier → skeleton overlay, class + confidence, saved video.
Extension: **Yoga Coach** – choose a target pose; the border turns green and a hold timer runs when the detected pose matches with > 80 % confidence.

![Pose training panel](Assets/pose_training_panel.png)
</details>

<details>
<summary><b>3 · Combined application — dashboard</b></summary>

`combined/combined_app.ipynb` loads both models at start-up, runs the pose model on every frame, runs the image model on a **background thread**, and shows a three-panel dashboard (Image · Pose · Summary), recording `combined/output/b4_combined.mp4`.
</details>

## 🧠 Concepts
Teachable Machine = **transfer learning by feature extraction**: a frozen pre-trained network (MobileNet / PoseNet) plus a small trained head. Full explanation, the link to the manual MobileNetV2 feature-extraction in Project A, and preprocessing details are in [CONCEPTS.md](CONCEPTS.md).


## 🛠️ Tools Used
TensorFlow / `tf_keras` · OpenCV · NumPy · Pillow · pandas · matplotlib · Jupyter · Teachable Machine · PoseNet (TF.js weights, MIT, via `@vladmandic/human-models`)

<details>
<summary><b>📊 Results & Limitations</b></summary>

**Results** – fill in after running the notebooks: Teachable Machine preview accuracy for both models, accuracy from the optional dataset-validation cells, and observed FPS in the combined dashboard.

**Limitations**
- Teachable Machine heads are small and trained for speed; the extractor stays frozen (no fine-tuning).
- The exported pose model needs **PoseNet** features (14,739 inputs), not MoveNet keypoints; PoseNet is re-implemented in Python from TF.js weights, so results should be validated with the dataset cell in `pose_classifier.ipynb`.
- Pose classification assumes one person with the whole body in view.
- No audio model was trained, so the combined app integrates two of the three Teachable Machine model types.
</details>
