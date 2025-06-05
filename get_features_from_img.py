import os
import cv2
import argparse
import numpy as np

from PIL import Image

import torch
from torchvision import transforms

from utils import get_feature
from ArcFace.mobile_model import mobileFaceNet
from mtcnn.src import detect_faces

def save_person_information_from_image(name, image_path):
    saved_model = './ArcFace/model/068.pth'
    info_path = os.path.join('./users', name)
    os.makedirs(info_path, exist_ok=True)
    model = mobileFaceNet()
    model.load_state_dict(torch.load(saved_model, map_location=torch.device('cpu'))['backbone_net_list'])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trans = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img = Image.open(image_path).convert('RGB')
    bboxes, _ = detect_faces(img)
    if len(bboxes) == 0:
        raise ValueError("No face detected in the image.")
    x1, y1, x2, y2 = map(int, bboxes[0][:4])
    img_np = np.array(img)
    person_img = img_np[y1:y2, x1:x2, :]
    person_img_pil = Image.fromarray(person_img)
    cropped_img_path = os.path.join(info_path, f'{name}.jpg')
    person_img_pil.save(cropped_img_path)
    feature = np.squeeze(get_feature(person_img, model, trans, device))
    np.savetxt(os.path.join(info_path, f'{name}.txt'), feature)
    print(f"Saved features for '{name}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help="User's name")
    parser.add_argument('--image', required=True, help="Path to image file (e.g. .jpg or .png)")
    args = parser.parse_args()

    save_person_information_from_image(args.name, args.image)
