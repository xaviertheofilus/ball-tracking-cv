# Movel AI Computer Vision Assignment

## Part A - Data Annotation Strategy

### Annotation approach

Use dense, frame-by-frame annotation. A fast ball can change direction suddenly after a bounce or hit, so interpolated keyframes may create incorrect positions between frames.

Annotate the ball with one tight bounding box using the class `ball`. Bounding boxes are practical for a non-CV vendor, easy to review, and directly compatible with YOLO. Keypoints are harder to place consistently on blurred balls, while segmentation masks add cost without enough benefit for this detector.

### Vendor instructions and edge cases

- Label every visible ball, including small and motion-blurred balls. The box should cover the visible blur, not the expected sharp ball size.
- For partial occlusion by a player or racket, box only the visible area and add an `occluded` attribute.
- Do not create a box when the ball is completely outside the frame. Mark the frame as `ball_not_visible`.
- If visual evidence suggests that the ball is present but its exact position cannot be placed reliably, do not guess. Mark it as `position_uncertain`.
- Ignore ball-like objects such as court markings, lights, logos, and balls held outside active play if they are outside the project scope.

`ball_not_visible` is a valid negative frame. `position_uncertain` means the frame should be reviewed or excluded from supervised training. Keeping them separate prevents uncertain guesses from becoming false ground truth and prevents valid negative frames from being discarded.

### Quality assurance

Before production, provide a short guideline with positive, negative, blurred, and occluded examples. Run a pilot batch and review it with the vendor. During production, double-label 10% of each batch, manually audit another random 10%, and track agreement, missed-ball rate, class errors, and box consistency. Review difficult cases weekly and update the shared examples when standards drift.

Reject a batch when more than 5% of audited frames contain a missed or false ball, labels use the wrong class, boxes consistently exclude visible pixels or include excessive background, status fields are misused, or duplicate/empty annotations indicate a tooling problem.

For the initial detector, I would choose two hours of high-quality dense footage. Accurate consecutive labels teach the detector ball appearance and motion transitions more effectively than sparse labels across ten hours. Broader footage should be added next to improve lighting, court, camera, and player diversity.

## Part B - Algorithm Development

### Video and approach

The submitted clip is `test2.mp4`: 25 seconds, 1,500 frames at 60 FPS, and 406 × 720 pixels. It is included directly in the repository.

I fine-tuned a pretrained YOLO11n model on the `ball` and `player` classes. The model was selected because it is small enough for CPU experimentation and produces bounding boxes and confidence scores directly. Training used 640-pixel input, batch size 4, and early stopping. The run was configured for 50 epochs and stopped after epoch 20 when validation stopped improving. Duplicate Roboflow exports were reduced to 595 unique training frames to keep CPU training practical.

At inference time, Ultralytics ByteTrack processes the video and the highest-confidence `ball` detection becomes the frame position at the bounding-box center. Frames without a valid ball detection are explicitly marked `missing`; I avoided interpolation because invented positions could hide real detector failures during evaluation.

The final validation results were approximately 0.59 mAP50 and 0.34 mAP50–95 across both classes. On the test video, the pipeline detected the ball in 486 of 1,500 frames (32.4% coverage), with a mean confidence of 0.59 for detected frames. The structured output is `outputs/test2_tracking/trajectory.csv`, and the reviewable overlay is `outputs/test2_tracking/test2.avi`.

### Strengths, limitations, and next steps

The pipeline is simple, reproducible, and produces both a visual result and frame-level data. It works best when the ball is large enough and visually distinct. Its main weakness is recall: the longest missing interval is frames 1175–1304, and performance drops when the ball is tiny, blurred, occluded, or unlike the training footage. The current confidence is also a detector score, not a calibrated probability.

Classical blob detection was considered but rejected because lighting, court lines, and player motion create unstable candidates. SAM-style segmentation was also rejected because it still needs reliable prompts and is too expensive for this CPU-focused baseline.

With more time, I would collect diverse color footage from the deployment camera, include more hard negatives and small-ball examples, split data by recording session rather than nearby frames, and evaluate ball-only precision and recall. I would then add a motion model such as a Kalman filter with gated recovery, while preserving `missing` when uncertainty is high.

Assumptions: the camera is fixed, only one active ball is tracked, and offline processing is acceptable for this prototype.

AI tools helped structure the notebook, diagnose path and environment errors, optimize CPU training, and edit this report. Dataset selection, labeling, training runs, output review, and final technical decisions were performed and verified by me.

## Part C - Diagnostic Reasoning

### Most likely root causes

1. Session and environment domain shift. Training and validation came from the same recording session, so the 78% score likely benefited from nearly identical lighting, background, camera angle, ball scale, and compression. Real-match footage changes all of these.
2. Non-independent validation split. Nearby frames can appear in both training and validation, making validation easier and overstating generalization.
3. Harder real-match motion and occlusion. Real play adds faster motion, more blur, player/racket occlusion, and a smaller apparent ball than controlled footage.

### How I would test the top cause

Create a small manually labeled evaluation set grouped by recording day and condition. Compare performance on the original session, a new session under similar lighting, and new sessions under different lighting. Break results down by ball size, brightness, blur, and occlusion. I would also inspect false negatives and detection-confidence distributions for each group. A large drop between sessions, especially within specific conditions, would confirm domain shift.

### One-week mitigation plan

- Days 1–2: collect and label short clips from the actual camera across expected lighting and match conditions; build a session-separated test set.
- Days 3–4: fine-tune with the new hard examples and balanced negatives, then tune the confidence threshold on the held-out sessions.
- Day 5: review failure clips, add targeted examples, and rerun the fixed evaluation.
- Days 6–7: freeze the model, produce the demo video, document supported conditions, and keep the previous model as a fallback.

I would tell the client that the current system is a prototype validated on limited conditions, not a guaranteed match-wide tracker. I would demonstrate known failure cases, report session-separated results, and agree on camera placement and lighting constraints for the demo.

### Continuous monitoring

Monitor rolling ball detection coverage: the percentage of frames with a ball detection above the chosen confidence threshold. Compare it with a baseline for the same camera and alert when it drops significantly for a sustained window. This is available without live ground truth and gives early warning of lighting, camera, or model-distribution changes.
