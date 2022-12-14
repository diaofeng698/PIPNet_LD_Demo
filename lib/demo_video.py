from math import sin, cos
import math
from functions import *
import data_utils
from networks import *
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import torch.nn.parallel
import torch.nn as nn
import torch
import time
from math import floor
import importlib
import pickle
import numpy as np
import cv2
import os
import sys
sys.path.insert(0, 'FaceBoxesV2')
sys.path.insert(0, '..')
from faceboxes_detector import *


# 3D face model (six points)
# Nose tip
# Chin
# Left eye left corner
# Right eye right corner
# Left Mouth corner
# Right mouth corner

face_coordination_in_real_world = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0]
], dtype=np.float64)


# Color
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
focal_length = 1368.933

# buffer and flag
ar_buffer = []
ar_buffer_size = 20
pose_buffer = []
pose_buffer_size = 20
yawn_buffer = []
yawn_buffer_size = 20
eye_gaze_buffer = []
eye_gaze_buffer_size = 5


# The Distance Matrix
dist_matrix = np.zeros((4, 1), dtype=np.float64)


def ar_calculate(frame, face_landmarks, eula_angle, det_w_h, det_x_y):
    det_width, det_height = det_w_h
    det_xmin, det_ymin = det_x_y
    point_sn = [[88, 92, 94, 90], [60, 64, 66, 62], [
        72, 68, 74, 70]]  # mouth, left eye, right_eye
    point_list = []
    pitch, yaw, roll = eula_angle
    pitch = pitch * np.pi / 180.0
    yaw = yaw * np.pi / 180.0
    roll = roll * np.pi / 180.0

    for list0 in point_sn:
        point_single = []
        for idx in list0:
            x, y = face_landmarks[2*idx] * det_width + \
                det_xmin, face_landmarks[1+2*idx] * det_height+det_ymin
            point_single.append([x, y])
            # cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 1)
        point_list.append(point_single)

    def ar_calc(points):
        p1, p2, p3, p4 = points
        width = math.sqrt((p1[0] - p2[0]) ** 2 +
                          (p1[1] - p2[1]) ** 2)  # / cos(abs(yaw))
        height = math.sqrt((p3[0] - p4[0]) ** 2 +
                           (p3[1] - p4[1]) ** 2)  # / cos(abs(pitch))
        return round(height / width, 3)

    mar = ar_calc(point_list[0])
    # Mirror the left and right eyes
    ear_r = ar_calc(point_list[1])
    ear_l = ar_calc(point_list[2])
    return mar, ear_l, ear_r

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotation_matrix_to_angles(rotation_matrix):
    """
    Calculate Euler angles from rotation matrix.
    :param rotation_matrix: A 3*3 matrix with the following structure
    [Cosz*Cosy  Cosz*Siny*Sinx - Sinz*Cosx  Cosz*Siny*Cosx + Sinz*Sinx]
    [Sinz*Cosy  Sinz*Siny*Sinx + Sinz*Cosx  Sinz*Siny*Cosx - Cosz*Sinx]
    [  -Siny             CosySinx                   Cosy*Cosx         ]
    :return: Angles in degrees for each axis
    """
    assert(isRotationMatrix(rotation_matrix))
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    if sy >= 1e-6:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0
    return np.array([x, y, z]) * 180. / math.pi


if not len(sys.argv) == 3:
    print('Format:')
    print('python lib/demo_video.py config_file video_file')
    exit(0)
experiment_name = sys.argv[1].split('/')[-1][:-3]
data_name = sys.argv[1].split('/')[-2]
config_path = '.experiments.{}.{}'.format(data_name, experiment_name)
video_file = sys.argv[2]

my_config = importlib.import_module(config_path, package='PIPNet')
Config = getattr(my_config, 'Config')
cfg = Config()
cfg.experiment_name = experiment_name
cfg.data_name = data_name

save_dir = os.path.join('./snapshots', cfg.data_name, cfg.experiment_name)

meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(
    os.path.join('data', cfg.data_name, 'meanface.txt'), cfg.num_nb)

if cfg.backbone == 'resnet18':
    resnet18 = models.resnet18(pretrained=cfg.pretrained)
    net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms,
                       input_size=cfg.input_size, net_stride=cfg.net_stride)
elif cfg.backbone == 'resnet50':
    resnet50 = models.resnet50(pretrained=cfg.pretrained)
    net = Pip_resnet50(resnet50, cfg.num_nb, num_lms=cfg.num_lms,
                       input_size=cfg.input_size, net_stride=cfg.net_stride)
elif cfg.backbone == 'resnet101':
    resnet101 = models.resnet101(pretrained=cfg.pretrained)
    net = Pip_resnet101(resnet101, cfg.num_nb, num_lms=cfg.num_lms,
                        input_size=cfg.input_size, net_stride=cfg.net_stride)
else:
    print('No such backbone!')
    exit(0)

if cfg.use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
net = net.to(device)

weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs-1))
state_dict = torch.load(weight_file, map_location=device)
net.load_state_dict(state_dict)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize(
    (cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])


def demo_video(video_file, net, preprocess, input_size, net_stride, num_nb, use_gpu, device):
    detector = FaceBoxesDetector(
        'FaceBoxes', 'FaceBoxesV2/weights/FaceBoxesV2.pth', use_gpu, device)
    my_thresh = 0.9
    det_box_scale = 1.2

    net.eval()

    if video_file == 'camera':
        cap = cv2.VideoCapture(0)
    else:
        # "/home/fdiao/Videos/test/sun0/DMS_20221024_sun0_xbx.mp4"
        cap = cv2.VideoCapture(video_file)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # set windoiw size
    cv2.namedWindow("demo",0)
    cv2.resizeWindow('demo', frame_width, frame_height)
    # write to file
    video_writer = cv2.VideoWriter('demovideo_record_man.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (frame_width, frame_height))

    count = 0
    while (cap.isOpened()):
        # start time
        start = time.time()

        ret, frame = cap.read()
        if ret == True:
            detections, _ = detector.detect(frame, my_thresh, 1)
            for i in range(len(detections)):
                det_xmin = detections[i][2]
                det_ymin = detections[i][3]
                det_width = detections[i][4]
                det_height = detections[i][5]
                det_xmax = det_xmin + det_width - 1
                det_ymax = det_ymin + det_height - 1

                det_xmin -= int(det_width * (det_box_scale-1)/2)
                # remove a part of top area for alignment, see paper for details
                det_ymin += int(det_height * (det_box_scale-1)/2)
                det_xmax += int(det_width * (det_box_scale-1)/2)
                det_ymax += int(det_height * (det_box_scale-1)/2)
                det_xmin = max(det_xmin, 0)
                det_ymin = max(det_ymin, 0)
                det_xmax = min(det_xmax, frame_width-1)
                det_ymax = min(det_ymax, frame_height-1)
                det_width = det_xmax - det_xmin + 1
                det_height = det_ymax - det_ymin + 1
                cv2.rectangle(frame, (det_xmin, det_ymin),
                              (det_xmax, det_ymax), (0, 0, 255), 2)
                det_crop = frame[det_ymin:det_ymax, det_xmin:det_xmax, :]
                det_crop = cv2.resize(det_crop, (input_size, input_size))
                inputs = Image.fromarray(
                    det_crop[:, :, ::-1].astype('uint8'), 'RGB')
                inputs = preprocess(inputs).unsqueeze(0)
                inputs = inputs.to(device)
                lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(
                    net, inputs, preprocess, input_size, net_stride, num_nb)
                lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
                tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(
                    cfg.num_lms, max_len)
                tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(
                    cfg.num_lms, max_len)
                tmp_x = torch.mean(
                    torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
                tmp_y = torch.mean(
                    torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
                lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
                lms_pred = lms_pred.cpu().numpy()
                lms_pred_merge = lms_pred_merge.cpu().numpy()
                for i in range(cfg.num_lms):
                    x_pred = lms_pred_merge[i*2] * det_width + det_xmin
                    y_pred = lms_pred_merge[i*2+1] * det_height + det_ymin
                    # cv2.circle(frame, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 1)
                    cv2.circle(frame, (int(x_pred), int(y_pred)), 1, (0, 0, 255), 1)
                    # print('x_pred: %f, y_pred: %f' % (x_pred, y_pred))

                # Use solvePnP function to get rotation vector
                face_coordination_in_image = []

                for idx in [54, 16, 60, 72, 76, 82]:
                # for idx in [54, 51, 76, 60, 82, 72]:
                    x, y = lms_pred_merge[2*idx] * det_width + \
                        det_xmin, lms_pred_merge[2*idx+1]*det_height+det_ymin
                    # cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 1)
                    face_coordination_in_image.append([x, y])
                face_coordination_in_image = np.array(
                    face_coordination_in_image, dtype=np.float64)

                cam_matrix = np.array(
                    [[focal_length, 0, frame_width / 2], [0, focal_length, frame_height / 2], [0, 0, 1]])
                success, rotation_vec, transition_vec = cv2.solvePnP(
                    face_coordination_in_real_world, face_coordination_in_image,
                    cam_matrix, dist_matrix)

                # Use Rodrigues function to convert rotation vector to matrix
                nose_end_point2D, jacobian = cv2.projectPoints(np.array(
                    [(0.0, 0.0, 100.0)]), rotation_vec, transition_vec, cam_matrix, dist_matrix)
                p1 = (int(lms_pred_merge[2*54] * det_width+det_xmin),
                      int(lms_pred_merge[2*54+1]*det_height+det_ymin))
                p2 = (int(nose_end_point2D[0][0][0]),
                      int(nose_end_point2D[0][0][1]))
                # line nose
                cv2.line(frame, p1, p2, (255, 0, 0), 2)

                # eula angle
                rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)
                # print(rotation_matrix)
                eula_angle = rotation_matrix_to_angles(rotation_matrix)
                print(eula_angle)

                # ar calculate
                ar = ar_calculate(frame, lms_pred_merge, eula_angle,
                                  (det_width, det_height), (det_xmin, det_ymin))

                pose_buffer.append(abs(eula_angle))
                ar_buffer.append(ar)
                left_eye = 0
                right_eye = 0
                if len(pose_buffer) > ar_buffer_size and len(ar_buffer) > ar_buffer_size:
                    pose_buffer.pop(0)
                    ar_buffer.pop(0)
                    ave_pitch, ave_yaw, _ = np.sum(
                        pose_buffer, axis=0) / pose_buffer_size
                    ave_mar, ave_lear, ave_rear = np.sum(
                        ar_buffer, axis=0) / ar_buffer_size

                    # EAR
                    cv2.putText(frame, "LEAR:"+str(round(ave_lear, 4)), (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
                    cv2.putText(frame, "REAR:"+str(round(ave_rear, 4)), (130, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
                    cv2.putText(frame, "MAR:"+str(round(ave_mar, 4)), (250, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)

                    if 0.8*ave_lear + 0.2*ave_rear < 0.22:
                        cv2.putText(frame, "Eye Closed", (10, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)

                    if ave_mar > 0.5:
                        cv2.putText(frame, "Yawn", (10, 210),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)

                    # Pose
                    cv2.putText(frame, "Pitch:"+str(round(ave_pitch, 4)), (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
                    cv2.putText(frame, "Yaw:"+str(round(ave_yaw, 4)), (160, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
                    if ave_pitch < 160 and eula_angle[0] > 0:
                        cv2.putText(frame, "Pose Up", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
                    elif ave_pitch < 160 and eula_angle[0] < 0:
                        cv2.putText(frame, "Pose Down", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
                    if ave_yaw-35 > 40 and eula_angle[1]-35>0:
                        cv2.putText(frame, "Pose Right", (10, 180),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
                    elif ave_yaw-35 > 30 and eula_angle[1]-35<0:
                        cv2.putText(frame, "Pose Left", (10, 180),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)

                # # eye gaze
                # # left eye 96
                # # right eye 97
                # left_eye_x, left_eye_y = lms_pred_merge[2*96] * det_width + \
                #     det_xmin, lms_pred_merge[1+2*96]*det_height+det_ymin
                # right_eye_x, right_eye_y = lms_pred_merge[2*97] * det_width + \
                #     det_xmin, lms_pred_merge[1+2*97]*det_height+det_ymin
                # mid_face_x, mid_face_y = lms_pred_merge[2*51] * det_width + \
                #     det_xmin, lms_pred_merge[1+2*51]*det_height+det_ymin
                # cv2.circle(frame, (int(left_eye_x), int(
                #     left_eye_y)), 1, (0, 0, 255), 1)
                # cv2.circle(frame, (int(right_eye_x), int(
                #     right_eye_y)), 1, (0, 0, 255), 1)
                # cv2.circle(frame, (int(mid_face_x), int(
                #     mid_face_y)), 1, (0, 0, 255), 1)

                # math.sqrt((left_eye_x-right_eye_x)**2+(left_eye_y-right_eye_y)**2)

                # eye_gaze_buffer.append(
                #     [(left_eye_x+right_eye_x)/2.0, mid_face_x])
                # print("left_eye_x",left_eye_x)
                # print("right_eye_x",right_eye_x)
                # print("mid_face_x",mid_face_x)

                # if len(eye_gaze_buffer) > eye_gaze_buffer_size:
                #     eye_gaze_buffer.pop(0)
                #     ave_eye_gaze, ave_mid_face = np.sum(
                #         eye_gaze_buffer, axis=0) / eye_gaze_buffer_size
                #     print("ave_eye_gaze: ", ave_eye_gaze,
                #           "ave_mid_face: ", ave_mid_face)
                #     if ave_eye_gaze - mid_face_x > -0.007*frame_width:
                #         cv2.putText(frame, "Look Right", (10, 270),
                #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
                #     elif ave_eye_gaze - mid_face_x < -0.01*frame_width:
                #         cv2.putText(frame, "Look Left", (10, 270),
                #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)

            count += 1
            #cv2.imwrite('video_out2/'+str(count)+'.jpg', frame)
            cv2.imshow('demo', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
            # end time
        end = time.time()
        fps = 1.0 / (end - start)
        print("FPS= %.2f" % (fps))
        video_writer.write(frame)

    cap.release()
    cv2.destroyAllWindows()


demo_video(video_file, net, preprocess, cfg.input_size,
           cfg.net_stride, cfg.num_nb, cfg.use_gpu, device)
