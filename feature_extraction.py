import cv2
import mediapipe as mp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from approximate_focal_length import get_focal_length
import torch
import csv
import os

class Position():
    def __init__(self, x = 0, y = 0, z = 0):
        self.x = x
        self.y = y
        self.z = z

video_path = './mediapipe_videos/videos/'
write_path = './mediapipe_videos/pose_estimate/'

files = len([name for name in os.listdir(video_path) if os.path.isfile(video_path + name)])
print('Find files:', files)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

average_frame_count = 10

for i in range(89, files + 1):
    cap = cv2.VideoCapture(video_path + 'shot_' + str(i) + '.mp4')
    out = cv2.VideoWriter(write_path + 'shot_pose_' + str(i) + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 360))

    ax = plt.axes(projection='3d')  # 用這個繪圖物件建立一個Axes物件(有3D座標)
    ax.set_xlabel('x label')
    ax.set_ylabel('z label')
    ax.set_zlabel('y label')  # 給三個座標軸註明

    camera_x = []
    camera_y = []
    camera_z = []

    object_positions = []
    camera_positions = []
    previous_2d = []
    average_camera_pos = Position()
    camera_position = Position()
    first = True
    correct = 0

    frame = 0
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as pose:
        print('\n### File: ' + str(i) + ' ###')
        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print("Correctness:", correct/frame)
                torch_camera_positions = torch.Tensor(camera_positions)
                # torch.save(torch.Tensor(object_positions), './mediapipe_videos/coordinates_object_pos/shot_' + str(i) + '.pt')
                torch.save(torch_camera_positions[1:], './mediapipe_videos/coordinates_camera_pos/shot_' + str(i) + '.pt')
                # print("Ignoring empty camera frame.")
                print(torch_camera_positions, torch_camera_positions.shape, torch_camera_positions[1:].shape)
                plt.savefig('./mediapipe_videos/coordinates_camera_pos/shot_' + str(i) + '.png')
                # plt.show()

                # with open('./mediapipe_videos/csv_object_pos/shot_' + str(i) + '.csv', 'w') as f:
                #     writer = csv.writer(f)
                #     writer.writerow(['X', 'Y', 'Z'])
                #     writer.writerows(object_positions)
                with open('./mediapipe_videos/csv_camera_pos/shot_' + str(i) + '.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    # writer.writerow(['X', 'Y', 'Z'])
                    writer.writerows(camera_positions[1:])
                # If loading a video, use 'break' instead of 'continue'.
                break


            # image = cv2.boxFilter(image, -1, (3, 3), normalize=1)
            # image = cv2.pyrMeanShiftFiltering(image, 10, 10,)  # 濾波

            # kernel = np.ones((3, 3), np.float32) / 25
            # image = cv2.filter2D(image, -1, kernel)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = pose.process(image)

            try:
                image_height, image_width, _ = image.shape
                # 取得 2D: pose_landmarks 和 3D: pose_world_landmarks 的取樣點
                nose_2d = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                nose_3d = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                left_shoulder_2d = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_shoulder_3d = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder_2d = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_shoulder_3d = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip_2d = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                left_hip_3d = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip_2d = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                right_hip_3d = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                left_elbow_2d = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                left_elbow_3d = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                right_elbow_2d = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                right_elbow_3d = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                left_eye_2d = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
                left_eye_3d = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
                right_eye_2d = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
                right_eye_3d = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]

                # Generates 2D and 3D matrix, respectively
                np_2d = np.array(
                    [[nose_2d.x * image_width, nose_2d.y * image_height],
                     [left_shoulder_2d.x * image_width, left_shoulder_2d.y * image_height],
                     [right_shoulder_2d.x * image_width, right_shoulder_2d.y * image_height],
                     [left_hip_2d.x * image_width, left_hip_2d.y * image_height],
                     [right_hip_2d.x * image_width, right_hip_2d.y * image_height],
                     [left_eye_2d.x * image_width, left_eye_2d.y * image_height],
                     [right_eye_2d.x * image_width, right_eye_2d.y * image_height],]
                )


                np_3d =  np.array(
                    [[nose_3d.x * image_width, nose_3d.y * image_height, nose_3d.z * image_width],
                    [left_shoulder_3d.x * image_width, left_shoulder_3d.y * image_height, left_shoulder_3d.z * image_width],
                     [right_shoulder_3d.x * image_width, right_shoulder_3d.y * image_height, right_shoulder_3d.z * image_width],
                     [left_hip_3d.x * image_width, left_hip_3d.y * image_height, left_hip_3d.z * image_width],
                     [right_hip_3d.x * image_width, right_hip_3d.y * image_height, right_hip_3d.z * image_width],
                     [left_eye_3d.x * image_width, left_eye_3d.y * image_height, left_eye_3d.z * image_width],
                     [right_eye_3d.x * image_width, right_eye_3d.y * image_height, right_eye_3d.z * image_width],]
                )

                # 取得 fx, fy
                fx, fy = get_focal_length(image_height, image_width)

                # 建立 Intrinsic Matrix
                intrinsic_matrix = np.array(
                    [[fx,  0,  image_width/2],
                     [ 0, fy, image_height/2],
                     [ 0,  0,              1]]
                )

                # 求 rotation 跟 translation 矩陣
                _, rotation, translation, _ = cv2.solvePnPRansac(np_3d, np_2d, intrinsic_matrix, None)

                R = np.array(cv2.Rodrigues(rotation)[0])
                T = np.array(translation)
                # print('R:\n', R, R.shape)
                # print('T:\n', T, T.shape)

                # RT = np.concatenate((R, translation), axis=1)
                # print(RT, RT.shape)

                # 求 camera 座標
                camera_pos = np.dot(-np.linalg.inv(R), T)
                camera_position = Position(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
                # camera_positions.append([camera_position.x, camera_position.y, camera_position.z])
                # print('Camera 3D:', camera_position.x, camera_position.y, camera_position.z)

                camera_x.append(camera_position.x)
                camera_y.append(camera_position.y)
                camera_z.append(camera_position.z)

                if frame % average_frame_count == 0:
                    frame_idx = int(frame/average_frame_count)
                    average_camera_pos.x = sum(list(camera_x[frame_idx : (frame_idx + average_frame_count)])) / average_frame_count
                    average_camera_pos.y = sum(list(camera_y[frame_idx : (frame_idx + average_frame_count)])) / average_frame_count
                    average_camera_pos.z = sum(list(camera_z[frame_idx : (frame_idx + average_frame_count)])) / average_frame_count
                    camera_positions.append((average_camera_pos.x, average_camera_pos.y, average_camera_pos.z))
                    print('Average position', average_camera_pos.x, average_camera_pos.y, average_camera_pos.z)

                # nose 在 3D(world space) 的位置
                n3d_1 = np.array([[nose_3d.x * image_width],
                                  [nose_3d.y * image_height],
                                  [nose_3d.z * image_width]]
                )
                # n3d_2 = np.array([[nose_3d.x * image_width],
                #                   [nose_3d.y * image_height],
                #                   [nose_3d.z * image_width],
                #                   [1.0]],
                # )

                # camera 在 3D(world space) 的位置
                camera_1 = np.array([[camera_position.x],
                                     [camera_position.y],
                                     [camera_position.z]]
                )

                # 從 3D(world space) 預測回 2D(image space) 的座標點
                (nose_end_point2D, jacobian) = cv2.projectPoints(n3d_1, rotation, translation, intrinsic_matrix, None)
                (camera_end_point2D, j) = cv2.projectPoints(camera_1, rotation, translation, intrinsic_matrix, None)
                # print('Camera 2D:', camera_end_point2D)

                # 3D(world space) 投影回去 2D(image space)上的座標
                nose_coordinates = [nose_end_point2D[0][0][0], nose_end_point2D[0][0][1]]
                camera_coordinates = [camera_end_point2D[0][0][0], camera_end_point2D[0][0][1]]

                '''
                convert world coordinates to camera coordinates
                object_pos = np.dot(RT, n3d_2)
                object_position = Position(object_pos[0][0].item(), object_pos[1][0].item(), object_pos[2][0].item())
                '''
                '''
                convert world coordinates to image coordinates [X, Y, 1]
                image_coord = np.dot(np.dot(intrinsic_matrix, RT), n3d_2)
                image_coord /= image_coord[2]
                '''

                # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS, pos = c)

                # 過濾
                # if first and (abs(nose_end_point2D[0][0][0] - nose_2d.x * image_width) <= 5 or abs(nose_end_point2D[0][0][1] - nose_2d.y * image_height) <= 5):
                if first and (abs(camera_coordinates[0] - image_width / 2) < 10) and (abs(camera_coordinates[1] - image_height / 2) < 10) \
                         and (abs(nose_coordinates[0] - nose_2d.x * image_width) < 10) and (abs(nose_coordinates[1] - nose_2d.y * image_height) < 10):
                    previous_3d_camera = Position(camera_position.x, camera_position.y, camera_position.z)
                    first = False

                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # mediapipe 的2D nose座標
                cv2.circle(image, (int(nose_2d.x * image_width), int(nose_2d.y * image_height)), 5, (255, 255, 255), 2)
                # 投影回去的2D nose座標
                cv2.circle(image, (int(nose_coordinates[0]), int(nose_coordinates[1])), 5, (255, 255, 0), 2)
                # camera 投影回去的2D座標
                cv2.circle(image, (int(camera_coordinates[0]), int(camera_coordinates[1])), 5, (255, 0, 255), -1)
                #  影像中心點
                cv2.circle(image, (int(image_width/2), int(image_height/2)), 5, (255, 255, 255), 2)
                # cv2.imshow('MediaPipe Pose', image)

                # if abs(nose_end_point2D[0][0][0] - nose_2d.x * image_width) > 15 or abs(nose_end_point2D[0][0][1] - nose_2d.y * image_height) > 15:
                if (abs(int(camera_end_point2D[0][0][0]) - image_width / 2) > 10) and (abs(int(camera_end_point2D[0][0][1]) - image_height / 2) > 10) \
                    and (abs(nose_coordinates[0] - nose_2d.x * image_width) > 10) and (abs(nose_coordinates[1] - nose_2d.y * image_height) > 10):
                    camera_x[-1] = previous_3d_camera.x
                    camera_y[-1] = previous_3d_camera.y
                    camera_z[-1] = previous_3d_camera.z
                    # print('# Refine #')
                else:
                    ax.scatter3D(average_camera_pos.x, average_camera_pos.z, average_camera_pos.y, color='Orange')
                    ax.scatter3D(0, 0, 0, color='Blue')
                    plt.draw()
                    plt.pause(0.001)
                    correct += 1

                previous_3d_camera = camera_position

            except Exception as e:
                print(e)
                pass

            # Write the output video
            out.write(image.astype('uint8'))
            # if cv2.waitKey(5) & 0xFF == 27:
            #     break
            frame += 1
    cap.release()
    out.release()
