import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from torchvision import models
import cv2

# -------------------- DEVICE --------------------
cuda_available = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda_available else "cpu")
print(f"CUDA Available: {cuda_available}")

# -------------------- MODEL LOADERS --------------------
def load_human_detector(weights_path):
    model = YOLO(weights_path)
    model.to(DEVICE)
    return model

def load_posture_model(weights_path, num_classes=2):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    for p in model.parameters():
        p.requires_grad = False
    return model

def load_clothing_model(weights_path, num_classes=3):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    for p in model.parameters():
        p.requires_grad = False
    return model

# -------------------- LOAD MODELS --------------------
human_detector = load_human_detector("runs/human_detection/weights/best.pt")
seg_model = YOLO("yolov8n-seg.pt").to(DEVICE)
posture_model = load_posture_model(
    "runs/posture_classification/resnet18_posture_classification_finetuned_best.pth"
)
clothing_model = load_clothing_model(
    "runs/clothing_classification/resnet50_clothing_classification_finetuned_best.pth"
)

# -------------------- TRANSFORMS --------------------
rgb_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

sil_transform = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# -------------------- UTILS --------------------
def box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter + 1e-6
    return inter / union

def draw_person_count(image: np.ndarray, count: int):
    """
    Draws number of detected persons on the top-left corner.
    """
    overlay = image.copy()

    text = f"Persons detected: {count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # Background rectangle
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(
        overlay,
        (10, 10),
        (10 + w + 10, 10 + h + 15),
        (0, 0, 0),
        -1
    )

    # Text
    cv2.putText(
        overlay,
        text,
        (15, 10 + h + 5),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA
    )

    return overlay


# -------------------- MAIN FUNCTION --------------------
def annotate_gif_with_models(
    input_gif_path,
    output_gif_path,
    posture_classes,
    clothing_classes,
    conf_thresh=0.3,
    iou_thresh=0.3,
):
    gif = Image.open(input_gif_path)
    output_frames = []
    duration = gif.info.get("duration", 100)

    for frame_idx in range(gif.n_frames):
        gif.seek(frame_idx)
        frame = gif.convert("RGB")
        frame_np = np.array(frame)
        draw = ImageDraw.Draw(frame)

        # --- Detection ---
        dets = human_detector(frame_np, conf=conf_thresh)[0]
        if dets.boxes is None:
            output_frames.append(frame)
            continue

        boxes = dets.boxes.xyxy.cpu().numpy()
        classes = dets.boxes.cls.cpu().numpy()

        # ✅ Person count (YOLO class 0 = person)
        person_count = sum(int(c) == 0 for c in classes)

        # --- Draw person count (top-left) ---
        count_text = f"Persons: {person_count}"
        font = ImageFont.load_default()

        text_w, text_h = draw.textbbox((0, 0), count_text, font=font)[2:]
        draw.rectangle(
            [5, 5, 5 + text_w + 10, 5 + text_h + 8],
            fill="black"
        )
        draw.text((10, 9), count_text, fill="lime", font=font)

        # --- Segmentation (ONCE per frame) ---
        seg = seg_model(frame_np)[0]
        masks = seg.masks.data.cpu().numpy() if seg.masks is not None else []

        for idx, box in enumerate(boxes):
            if int(classes[idx]) != 0:
                continue

            x1, y1, x2, y2 = map(int, box)
            crop_rgb = frame_np[y1:y2, x1:x2]
            if crop_rgb.size == 0:
                continue

            # --- Match segmentation mask ---
            best_iou, best_mask = 0, None
            for m_box, mask in zip(seg.boxes.xyxy.cpu().numpy(), masks):
                iou = box_iou(box, m_box)
                if iou > best_iou:
                    best_iou, best_mask = iou, mask

            if best_mask is None or best_iou < iou_thresh:
                continue

            sil = np.where(best_mask > 0.5, 0, 255).astype(np.uint8)
            sil_crop = sil[y1:y2, x1:x2]
            sil_img = Image.fromarray(sil_crop)

            posture_tensor = sil_transform(sil_img).unsqueeze(0).to(DEVICE)
            clothing_tensor = rgb_transform(
                Image.fromarray(crop_rgb)
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                posture_idx = posture_model(posture_tensor).argmax(1).item()
                clothing_idx = clothing_model(clothing_tensor).argmax(1).item()

            label = f"{posture_classes[posture_idx]} | {clothing_classes[clothing_idx]}"

            # --- Draw box + label ---
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.rectangle(
                [x1, y1 - 14, x1 + len(label) * 7, y1],
                fill="red"
            )
            draw.text((x1, y1 - 14), label, fill="white", font=font)

        output_frames.append(frame)

    output_frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=output_frames[1:],
        duration=duration,
        loop=0,
    )


# -------------------- RUN --------------------
annotate_gif_with_models(
    input_gif_path="demo/problem.gif",
    output_gif_path="demo/problem_annotated.gif",
    posture_classes=["sitting", "standing"],
    clothing_classes=["light", "medium", "heavy"],
)
