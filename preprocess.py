import os
import argparse
from tensorflow.keras.applications.vgg16 import VGG16

from utils.convert import *

parser = argparse.ArgumentParser()

parser.add_argument("-f", "--frame_rate", default = 6, help = "n 프레임 당 한 이미지 저장 (낮을수록 학습량 증가)")
parser.add_argument("-l", "--lstm", default = 15, help = "LSTM 간격 (낮을수록 학습량 증가)")
parser.add_argument("-i", "--img_size", default = 112, help = "이미지 Resize 크기 (높을수록 학습량 n^2배 증가)")

src = '학습영상(표준운동)39가지/' # 데이터셋 폴더 명


args = parser.parse_args()

FRAME_PER_RATE = args.frame_rate
IMG_SIZE = args.img_size
DEVIDER = args.lstm

framelist, datalist, actions = video_to_frame(src, FRAME_PER_RATE, IMG_SIZE, DEVIDER)
base_model = VGG16(weights='imagenet', include_top=False)
X_train, X_data, y_data = extract_features(base_model, framelist, datalist, actions, DEVIDER)

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