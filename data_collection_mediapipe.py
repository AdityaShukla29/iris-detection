# 1. Install Dependencies and Setup

# 2. Collect Images
import os
import time
import uuid
import cv2
import mediapipe as mp
import json
import numpy as np

IMAGES_PATH = 'data'
LABELS_PATH = 'data/labels'
number_images = 600

os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(LABELS_PATH, exist_ok=True)

cap = cv2.VideoCapture(0)
for imgnum in range(number_images):
    print('Collecting image {}'.format(imgnum))
    ret, frame = cap.read()
    imgname = os.path.join(IMAGES_PATH, f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)
    time.sleep(0.5)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# 3. Auto-Annotate Eyes using MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

def get_eye_center(landmarks, eye_indices, image_width, image_height):
    eye_points = []
    for idx in eye_indices:
        x = landmarks[idx].x * image_width
        y = landmarks[idx].y * image_height
        eye_points.append([x, y])
    eye_points = np.array(eye_points)
    center_x = np.mean(eye_points[:, 0])
    center_y = np.mean(eye_points[:, 1])
    return center_x, center_y

successful_annotations = 0
failed_annotations = 0

for image_file in os.listdir(IMAGES_PATH):
    if image_file.endswith('.jpg'):
        image_path = os.path.join(IMAGES_PATH, image_file)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = image.shape
                left_eye_x, left_eye_y = get_eye_center(face_landmarks.landmark, LEFT_EYE_INDICES, w, h)
                right_eye_x, right_eye_y = get_eye_center(face_landmarks.landmark, RIGHT_EYE_INDICES, w, h)
                annotation = {
                    'image': image_file,
                    'class': [1, 1],
                    'keypoints': [left_eye_x, left_eye_y, right_eye_x, right_eye_y],
                    'shapes': [
                        {'label': 'LeftEye', 'points': [[left_eye_x, left_eye_y]]},
                        {'label': 'RightEye', 'points': [[right_eye_x, right_eye_y]]}
                    ]
                }
                label_file = image_file.replace('.jpg', '.json')
                label_path = os.path.join(LABELS_PATH, label_file)
                with open(label_path, 'w') as f:
                    json.dump(annotation, f)
                successful_annotations += 1
                print(f"Annotated: {image_file}")
                break
        else:
            print(f"No face detected in: {image_file}")
            failed_annotations += 1

print(f"\nAnnotation Complete!")
print(f"Successful: {successful_annotations}")
print(f"Failed: {failed_annotations}")

# 4. Verify Annotations (Optional)
import matplotlib.pyplot as plt

sample_images = [f for f in os.listdir(IMAGES_PATH) if f.endswith('.jpg')][:4]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, image_file in enumerate(sample_images):
    image_path = os.path.join(IMAGES_PATH, image_file)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label_file = image_file.replace('.jpg', '.json')
    label_path = os.path.join(LABELS_PATH, label_file)
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            annotation = json.load(f)
        keypoints = annotation['keypoints']
        cv2.circle(image_rgb, (int(keypoints[0]), int(keypoints[1])), 5, (255, 0, 0), -1)
        cv2.circle(image_rgb, (int(keypoints[2]), int(keypoints[3])), 5, (0, 255, 0), -1)
    axes[i].imshow(image_rgb)
    axes[i].set_title(f'Image {i+1}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# 5. Split Data into Train/Test/Val
import shutil
import random

for folder in ['train', 'test', 'val']:
    os.makedirs(os.path.join('data', folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join('data', folder, 'labels'), exist_ok=True)

annotated_images = []
for image_file in os.listdir(IMAGES_PATH):
    if image_file.endswith('.jpg'):
        label_file = image_file.replace('.jpg', '.json')
        if os.path.exists(os.path.join(LABELS_PATH, label_file)):
            annotated_images.append(image_file)

print(f"Total annotated images: {len(annotated_images)}")

random.shuffle(annotated_images)
train_split = int(0.7 * len(annotated_images))
val_split = int(0.85 * len(annotated_images))

train_images = annotated_images[:train_split]
val_images = annotated_images[train_split:val_split]
test_images = annotated_images[val_split:]

print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

for split, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
    for image_file in images:
        src_image = os.path.join(IMAGES_PATH, image_file)
        dst_image = os.path.join('data', split, 'images', image_file)
        shutil.copy2(src_image, dst_image)
        label_file = image_file.replace('.jpg', '.json')
        src_label = os.path.join(LABELS_PATH, label_file)
        dst_label = os.path.join('data', split, 'labels', label_file)
        shutil.copy2(src_label, dst_label)

print("Data split complete!")

# 6. Image Augmentation
import albumentations as alb

for folder in ['train', 'test', 'val']:
    os.makedirs(os.path.join('aug_data', folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join('aug_data', folder, 'labels'), exist_ok=True)

augmentor = alb.Compose([
    alb.RandomCrop(width=450, height=450), 
    alb.HorizontalFlip(p=0.5), 
    alb.RandomBrightnessContrast(p=0.2),
    alb.RandomGamma(p=0.2), 
    alb.RGBShift(p=0.2), 
    alb.VerticalFlip(p=0.5)
], keypoint_params=alb.KeypointParams(format='xy', label_fields=['class_labels']))

for partition in ['train', 'test', 'val']: 
    for image_file in os.listdir(os.path.join('data', partition, 'images')):
        if image_file.endswith('.jpg'):
            img = cv2.imread(os.path.join('data', partition, 'images', image_file))
            label_file = image_file.replace('.jpg', '.json')
            label_path = os.path.join('data', partition, 'labels', label_file)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label = json.load(f)
                keypoints = label['keypoints']
                left_eye = (keypoints[0], keypoints[1])
                right_eye = (keypoints[2], keypoints[3])
                h, w = img.shape[:2]
                coords = [keypoints[0]/w, keypoints[1]/h, keypoints[2]/w, keypoints[3]/h]
                try:
                    for x in range(6):
                        keypoints_list = [left_eye, right_eye]
                        augmented = augmentor(image=img, keypoints=keypoints_list, class_labels=['LeftEye', 'RightEye'])
                        aug_image_path = os.path.join('aug_data', partition, 'images', f'{image_file.split(".")[0]}.{x}.jpg')
                        cv2.imwrite(aug_image_path, augmented['image'])
                        annotation = {
                            'image': f'{image_file.split(".")[0]}.{x}.jpg',
                            'class': [0, 0],
                            'keypoints': [0, 0, 0, 0]
                        }
                        if len(augmented['keypoints']) > 0:
                            for idx, cl in enumerate(augmented['class_labels']):
                                if cl == 'LeftEye':
                                    annotation['class'][0] = 1
                                    annotation['keypoints'][0] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][1] = augmented['keypoints'][idx][1]
                                if cl == 'RightEye':
                                    annotation['class'][1] = 1
                                    annotation['keypoints'][2] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][3] = augmented['keypoints'][idx][1]
                        annotation['keypoints'] = list(np.divide(annotation['keypoints'], [450, 450, 450, 450]))
                        aug_label_path = os.path.join('aug_data', partition, 'labels', f'{image_file.split(".")[0]}.{x}.json')
                        with open(aug_label_path, 'w') as f:
                            json.dump(annotation, f)
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
    print(f"Augmentation complete for {partition}")

print("All augmentation complete!")

# 7. Dataset Summary
total_images = 0
for partition in ['train', 'test', 'val']:
    count = len([f for f in os.listdir(os.path.join('aug_data', partition, 'images')) if f.endswith('.jpg')])
    print(f"{partition.capitalize()}: {count} images")
    total_images += count

print(f"\nTotal augmented dataset: {total_images} images")
print(f"Ready for training!")