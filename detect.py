import os
import cv2
import time
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import transforms

from PIL import Image, ImageFont
from picamera2 import Picamera2
from libcamera import controls

from dnet import Darknet
from mtcnn.src import detect_faces
from kalman_filter.tracker import Tracker
from ArcFace.mobile_model import mobileFaceNet

from utils import cosin_metric, get_feature, draw_ch_zn, load_classes, write_results

font = ImageFont.truetype('simhei.ttf', 20, encoding='utf-8')
cfgfile = "cfg/yolov3.cfg"
weightsfile = "weights/yolov3.weights"
classes = load_classes('data/names.names')
confidence = 0.25
nms_thesh = 0.4
CUDA = torch.cuda.is_available()
inp_dim = 160 

model = Darknet(cfgfile)
model.load_weights(weightsfile)
model.net_info["height"] = str(inp_dim)
assert inp_dim % 32 == 0
assert inp_dim > 32

if CUDA:
    model.cuda()

model.eval()

tracker = Tracker(dist_thresh=160, max_frames_to_skip=100, max_trace_length=5, trackIdCount=1)

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": 'RGB888', "size": (640, 480)}))
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
picam2.start()
time.sleep(1.0)

saved_model = './ArcFace/model/068.pth'
name_list = os.listdir('./users')
path_list = [os.path.join('./users', i, f'{i}.txt') for i in name_list]
total_features = np.empty((0, 128), np.float32)
for path in path_list:
    temp = np.loadtxt(path)
    total_features = np.vstack((total_features, temp))
threshold = 0.5
model_facenet = mobileFaceNet()
model_facenet.load_state_dict(torch.load(saved_model, map_location=torch.device('cpu'))['backbone_net_list'])
model_facenet.eval()
device = torch.device("cuda" if CUDA else "cpu")
model_facenet.to(device)
trans = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def prep_image(img, inp_dim):
    img_resized = cv2.resize(img, (inp_dim, inp_dim))
    img_rgb = img_resized[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_tensor = torch.from_numpy(img_rgb).float().div(255.0).unsqueeze(0)
    return img_tensor, img, (img.shape[1], img.shape[0])

def write(x, img):
    if np.isnan(x[1:5]).any() or np.isinf(x[1:5]).any():
        return img
    c1 = tuple(x[1:3].astype(int))
    c2 = tuple(x[3:5].astype(int))
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = (255, 0, 0)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2_label = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2_label, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img

def select_person(output):
    return [i for i in output if i[-1] == 0]

def to_xy(outputs):
    return [[[0.5*(o[1]+o[3])], [0.5*(o[2]+o[4])]] for o in outputs]

def xy_to_normal(outputs, tracks):
    output_normal = []
    for i, output in enumerate(outputs):
        x_center, y_center = tracks[i].prediction[0], tracks[i].prediction[1]
        width = output[3] - output[1]
        height = output[4] - output[2]
        x_l = int(x_center - 0.5 * width)
        y_l = int(y_center - 0.5 * height)
        x_r = int(x_center + 0.5 * width)
        y_r = int(y_center + 0.5 * height)
        track_id = tracks[i].track_id
        output_normal.append([x_l, y_l, x_r, y_r, track_id])
    return output_normal

confirm = False
name = ""
count_yolo = 0

while True:
    start_time = time.time()
    color_image = picam2.capture_array()
    img_tensor, orig_im, dim = prep_image(color_image, inp_dim)
    im_dim = torch.FloatTensor(dim).repeat(1, 2)
    if CUDA:
        im_dim = im_dim.cuda()
        img_tensor = img_tensor.cuda()

    if count_yolo % 3 == 0:
        output = model(Variable(img_tensor), CUDA)
        output = write_results(output, confidence, 80, nms=True, nms_conf=nms_thesh)
        if isinstance(output, int):
            fps = 1.0 / (time.time() - start_time)
            print(f"fps= {fps:.2f}")
            cv2.imshow("frame", orig_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim
        output[:, [1, 3]] *= color_image.shape[1]
        output[:, [2, 4]] *= color_image.shape[0]
        output = output.cpu().numpy()
        output = select_person(output)
        output = np.array(output)
        output_update = output
    else:
        output = output_update
    count_yolo += 1
    list(map(lambda x: write(x, orig_im), output))

    output_kalman_xywh = to_xy(output)
    if output_kalman_xywh:
        tracker.Update(output_kalman_xywh)
    outputs_kalman_normal = np.array(xy_to_normal(output, tracker.tracks))
    for output_kalman_normal in outputs_kalman_normal:
        cv2.rectangle(orig_im, (int(output_kalman_normal[0]), int(output_kalman_normal[1])), (int(output_kalman_normal[2]), int(output_kalman_normal[3])), (255, 255, 255), 2)
        cv2.putText(orig_im, str(output_kalman_normal[4]), (int(output_kalman_normal[0]), int(output_kalman_normal[1])), 0, 0.5, (0, 255, 0), 2)

    if not confirm:
        img_pil = Image.fromarray(color_image)
        bboxes, _ = detect_faces(img_pil)
        if len(bboxes) == 0:
            print('No face detected')
        else:
            for bbox in bboxes:
                x1 = max(0, int(bbox[0]))
                y1 = max(0, int(bbox[1]))
                x2 = min(color_image.shape[1], int(bbox[2]))
                y2 = min(color_image.shape[0], int(bbox[3]))
                if x2 <= x1 or y2 <= y1:
                    continue
                loc_x_y = [x2, y1]
                person_img = color_image[y1:y2, x1:x2].copy()
                feature = np.squeeze(get_feature(person_img, model_facenet, trans, device))
                cos_distance = cosin_metric(total_features, feature)
                index = np.argmax(cos_distance)
                if cos_distance[index] <= threshold:
                    continue
                name = name_list[index]
                orig_im = draw_ch_zn(orig_im, name, font, loc_x_y)
                cv2.rectangle(orig_im, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if output is not None and len(output) > 0:
        label_pos = (int(output[0][1]) + 100, int(output[0][2]) + 20)
        cv2.putText(orig_im, f'{name}', label_pos, cv2.FONT_HERSHEY_PLAIN, 2, [0, 255, 0], 2)
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(orig_im, f'FPS: {fps:.2f}', (10, 450), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
    cv2.imshow("", orig_im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()

