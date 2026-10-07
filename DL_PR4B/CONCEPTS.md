# CONCEPTS — How Teachable Machine Works (and How It Maps to This Project)

## 1. What Teachable Machine does internally
Teachable Machine does **not** train a deep network from scratch in the browser. It takes a **pre-trained feature extractor** and trains only a **small neural-network head** on top of it, using the samples you record:

| Project type | Frozen pre-trained extractor | Trained head (your classes) | Model input |
|---|---|---|---|
| Image (vehicles, 20 classes) | MobileNetV2-style CNN (`Conv_1`, `out_relu`, global average pooling) | `Dense(100) → Dense(20)` — **130,100** parameters | 224×224×3 image scaled to [-1, 1] |
| Pose (yoga, 5 poses) | PoseNet, MobileNetV1 ×0.75, output stride 16 | `Dense(100, relu) → Dropout(0.5) → Dense(5, softmax)` — **1,474,500** parameters | 14,739 numbers (see below) |

This is **Transfer Learning via Feature Extraction**.

## 2. Why it works
The extractor was pre-trained on a very large dataset, so its convolutional layers already know edges, textures, shapes and object parts. Only the last classification layers need to learn *which combination of those features* means "Bus" or "Tree pose". That is why a few dozen webcam samples per class and a couple of minutes of training are enough.

## 3. Same idea as Project A, Task A6.2 (Feature Extraction)
In Project A, `MobileNetV2(include_top=False, weights='imagenet')` was frozen, a new `GlobalAveragePooling2D → Dense(128) → Dense(10)` head was added, and only the head was trained. Teachable Machine does exactly the same thing: **one is done visually in the browser, the other is written in code**. Fine-tuning (Task A6.4) is the step Teachable Machine does *not* take — the base stays frozen — which is one reason TM models are fast but not maximally accurate.

## 4. Why preprocessing matters
The head was trained on features computed from images prepared in one specific way. At deployment the same transform must be applied or the features shift and predictions degrade:
- **Image model:** centre-crop to a square, resize to **224×224**, scale pixels with `x / 127.5 − 1` → range **[-1, 1]**.
- **Pose model:** pad the frame to a square, resize to **257×257**, scale to [-1, 1], run PoseNet, apply a sigmoid to the heat-map, then flatten heat-maps and offsets.

## 5. How PoseNet feeds the pose model
PoseNet is a CNN that outputs, on a 17×17 grid, one **heat-map** per body keypoint (17 channels: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) and **offset vectors** (34 channels) that refine each location. Teachable Machine concatenates them: 17 × 17 × (17 + 34) = **14,739** values, which is exactly the input size of the exported pose head (`models/model.json`). The classifier therefore looks at *where the body parts are*, not at pixel colours, which makes it largely independent of lighting and clothing. The skeleton drawn on screen is decoded from the same heat-maps (arg-max cell + offset per keypoint).

> The project brief mentions MoveNet `(1, 17, 2)` keypoints. A model exported from Teachable Machine's Pose project cannot take that input; PoseNet features are required, which is what `pose/pose_classifier.ipynb` uses.

## 6. Why a "background" class matters
A softmax classifier must always pick one of its classes. Without a class for "nothing relevant" (empty room, silence, no person) it will output a confident but wrong label. A background class gives the model a legitimate answer for the idle case. *(An audio model was not part of this submission, but the same reasoning applies to the image model's "no object" case.)*

## 7. Why threading in the combined app
Camera frames arrive ~30 times a second. The pose pipeline (PoseNet + head) runs every frame, while the image classifier is slower and does not need to run at the same rate. In `combined/combined_app.ipynb` the image model runs on a **background thread** and publishes its latest result; the main loop never waits for it, so the video stays smooth. (In the brief, this role is played by the audio thread.)

## 8. Limitations
- Teachable Machine models are small and tuned for speed, not maximum accuracy; for production, fine-tune a larger model as in Project A (Task A6.4).
- Accuracy depends on how varied the training samples were (angle, distance, lighting, background).
- Pose classes assume the whole body is visible and a single person is in frame.
- Model-introspection numbers for this project are printed live by the notebooks (`model.summary()`, `count_params()`).
