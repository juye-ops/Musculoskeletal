import os
import argparse
import json

from utils.convert import *

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", default = "dataset/학습영상(표준운동)39가지/", type=str, help = "데이터셋 폴더 명")
parser.add_argument("-f", "--frame_rate", default = 6, type=int, help = "n 프레임 당 한 이미지 저장 (낮을수록 학습량 증가)")
parser.add_argument("-l", "--lstm", default = 15, type=int, help = "LSTM 간격 (낮을수록 학습량 증가)")
parser.add_argument("-i", "--img_size", default = 112, type=int, help = "이미지 Resize 크기 (높을수록 학습량 n^2배 증가)")


args = parser.parse_args()

dataset = args.dataset
FRAME_PER_RATE = args.frame_rate
IMG_SIZE = args.img_size
DEVIDER = args.lstm

framelist, datalist, actions = video_to_frame(dataset, FRAME_PER_RATE, IMG_SIZE, DEVIDER)
X_train, X_data, y_data, feature_size = extract_features(framelist, datalist, actions, DEVIDER)

from sklearn.model_selection import train_test_split
X_data=X_data.astype(np.float32)
y_data=y_data.astype(np.float32)
print(X_train.shape,y_data.shape)
print(X_data.shape,y_data.shape)
xf_train, xf_val, yf_train, yf_val = train_test_split(X_train, y_data, test_size=0.2,random_state=2021, shuffle=True) # 이미지셋 분리
xj_train, xj_val, yj_train, yj_val = train_test_split(X_data, y_data, test_size=0.2,random_state=2021, shuffle=True)  # 스켈레톤 분리


xf_val,xf_test,yf_val ,yf_test = train_test_split(xf_val,yf_val, test_size=0.5,random_state=2022, shuffle=True) # 이미지셋 V:T 분리
xj_val,xj_test,yj_val ,yj_test = train_test_split(xj_val,yj_val, test_size=0.5,random_state=2022, shuffle=True) # 스켈레톤 V:T 분리

print('-----')
print(xf_train.shape, yf_train.shape)
print(xf_val.shape, yf_val.shape)
print(xf_test.shape, yf_test.shape)
print()
print(xj_train.shape,yj_train.shape)
print(xj_val.shape, yj_val.shape)
print(xj_test.shape, yj_test.shape)
print('-----')

np.save("dataset/xf_train.npy", xf_train)
np.save("dataset/yf_train.npy", yf_train)
np.save("dataset/xf_val.npy", xf_val)
np.save("dataset/yf_val.npy", yf_val)

np.save("dataset/xj_train.npy", xj_train)
np.save("dataset/yj_train.npy", yj_train)
np.save("dataset/xj_val.npy", xj_val)
np.save("dataset/yj_val.npy", yj_val)

f = open("dataset/configuration.json", "w")


json.dump({"shape": (xf_train.shape[1:3], xj_train.shape[1:3]), "FRAME_PER_RATE": FRAME_PER_RATE, "IMG_SIZE": IMG_SIZE, "DEVIDER": DEVIDER, "feature_size": feature_size}, f)