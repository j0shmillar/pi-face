import os
import cv2
import time
import argparse
import numpy as np

from PIL import Image
from picamera2 import Picamera2

import torch
from torchvision import transforms

from utils import get_feature
from ArcFace.mobile_model import mobileFaceNet
from mtcnn.src import detect_faces, show_bboxes

def save_person_information(name):
    saved_model = './ArcFace/model/068.pth'
    info_path = './users/' + name
    if not os.path.exists(info_path):
        os.makedirs(info_path)

    model = mobileFaceNet()
    model.load_state_dict(torch.load(saved_model, map_location=torch.device('cpu'))['backbone_net_list'])
    model.eval()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)

    trans = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    cam = Picamera2()
    cam.preview_configuration.main.size = (640, 480)
    cam.preview_configuration.main.format = "RGB888"
    cam.configure("preview")
    cam.start()
    time.sleep(1)

    print("Press 'c' to capture, 'q' to quit")

    while True:
        frame = cam.capture_array() 
        img = Image.fromarray(frame)
        bboxes, landmark = detect_faces(img)
        if len(bboxes) > 0:
            show_img = show_bboxes(img, bboxes, landmark)
            show_img = np.array(show_img)[:, :, ::-1]  
            show_img = show_img.copy()
            cv2.putText(show_img, "press 'c' to crop your face", (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, [255, 0, 0], 2)
            cv2.imshow('img', show_img)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            if len(bboxes) > 0:
                x1, y1, x2, y2 = map(int, bboxes[0][:4])
                person_img = frame[y1:y2, x1:x2, :]
                cv2.imshow('crop', person_img[:, :, ::-1])
                cv2.imwrite(os.path.join(info_path, f'{name}.jpg'), person_img[:, :, ::-1])
                feature = np.squeeze(get_feature(person_img, model, trans, device))
                np.savetxt(os.path.join(info_path, f'{name}.txt'), feature)
            else:
                print("No face detected")
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-n', '--name', default=None, help="user's name")
    arg = parse.parse_args()
    name = arg.name
    if name is None:
        raise ValueError("input with --name 'your_name'")
    save_person_information(name)
