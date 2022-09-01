import os
import cv2
import json
import numpy as np

def video_to_frame(src, FRAME_PER_RATE, IMG_SIZE, DEVIDER):
    alist = []
    datalist = []
    framelist = []
    actions = []

    directory = os.listdir(src)

    for i in directory:
        label = 0
        name = os.listdir(src + i)
        print(i.split(" ")[0] + "\r")
        for j in name:
            skel = os.listdir(src + i + '/' + j)
            for k in skel:
                if k.endswith('mp4'):
                    cap = cv2.VideoCapture(src + i + '/' + j + '/' + k)
                    x = 0
                    while (cap.isOpened()):  # 영상 캡쳐링 진행
                        ret, frame = cap.read()
                        if (ret != True):  # 종료 조건
                            break
                        if not x % FRAME_PER_RATE:  # Frame이 FRAME_PER_RATE 마다 한 장 저장
                            frame = cv2.resize(frame, dsize=(IMG_SIZE, IMG_SIZE))
                            framelist.append(frame)
                        x += 1

                if 'json' in k:
                    with open(src + i + '/' + j + '/' + k, 'r') as f:
                        json_data = json.load(f)

                        try:
                            x = 0
                            for data in json_data['infos']:
                                xyzList = []

                                if not x % FRAME_PER_RATE:
                                    for xyz in data['positions']:
                                        xyzList.append(xyz['x'])
                                        xyzList.append(xyz['y'])
                                        xyzList.append(xyz['z'])
                                    xyzList.append(label)

                                    datalist.append(xyzList)
                                x += 1
                        except:
                            print(i, j, k)

            ##### LSTM 분리 시 DEVIDER에 나누어 떨어지는 크기로 저장(나머지는 버림)######

            sliceIdx = min(len(framelist), len(datalist))
            sliceIdx -= (sliceIdx % DEVIDER)
            if sliceIdx % DEVIDER:
                framelist = framelist[:sliceIdx]
                datalist = datalist[:sliceIdx]
            elif not sliceIdx % DEVIDER:
                framelist = framelist[:sliceIdx]
                datalist = datalist[:sliceIdx]

            ###########################################################################

            label += 1
            if not j in actions:
                actions.append(j)

    framelist = np.array(framelist)
    datalist = np.array(datalist)

    return framelist, datalist, actions

def extract_features(base_model, framelist, datalist, actions, DEVIDER):
    X_train = base_model.predict(framelist)  # *** 특징 추출을 통한 경량화 ***

    t = X_train.shape
    feature_size = t[1] * t[2] * t[3]
    X_train = X_train.reshape(-1, DEVIDER, feature_size)  # *** 이미지: LSTM을 이용하기 위한 reshape ***
    x_data = datalist.reshape(-1, DEVIDER, 97)  # *** 스켈레톤: LSTM을 이용하기 위한 reshape ***

    X_data = x_data[:, :, :-1]
    labels = x_data[:, 0, -1]
    print(X_train.shape)
    print(X_data.shape)
    print(labels)

    print(actions)

    from tensorflow.keras.utils import to_categorical
    y_data = to_categorical(labels, num_classes=len(actions))

    return X_train, X_data, y_data