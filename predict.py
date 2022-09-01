import os

from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, concatenate, Dropout
import numpy as np
import cv2
import json

f = open("dataset/configuration.json", "r")
meta = json.load(f)

(xf_shape, xj_shape), FRAME_PER_RATE, IMG_SIZE, DEVIDER, feature_size = list(meta.values())

dir = os.listdir("dataset/testset")
if len(dir) == 2:
    if dir[0].endswith(".json") and dir[1].endswith(".mp4"):
        video = f"dataset/testset/{dir[1]}"
        skeleton = f"dataset/testset/{dir[0]}"
    elif dir[0].endswith(".mp4") and dir[1].endswith(".json"):
        video = f"dataset/testset/{dir[0]}"
        skeleton = f"dataset/testset/{dir[1]}"
    else:
        raise OSError
else:
    raise OSError

input1 = keras.Input(shape=xf_shape)  # 이미지 레이어
input2 = keras.Input(shape=xj_shape)  # 스켈레톤 레이어

# 이미지 레이어
x = LSTM(512, activation='relu')(input1)
x = Dropout(0.025)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.025)(x)
x = Dense(64, activation='relu')(x)
m1 = keras.Model(input1, x)


# 스켈레톤 레이어
y = LSTM(128, activation='relu')(input2)
y = Dense(64, activation='relu')(y)
m2 = keras.Model(input2, y)

# 레이어 결합
z = concatenate([m1.output, m2.output])

outputs = Dense(39, activation='softmax')(z)

# 멀티 인풋 모델 생성
model = keras.Model([input1, input2], outputs)

model.load_weights("model.h5") # !!위험!!최고 가중치 추출: 마지막 기록학습이 날아갑니다.

frames = []
xyzs = []
# 같은 전처리 과정 후 위에서 설정한 영상 및 json파일로 predict 후 영상에 대한 결과 검출
cap = cv2.VideoCapture(video)
while (cap.isOpened()):
    ret, frame = cap.read()
    if (ret != True):
        break
    frame = cv2.resize(frame, dsize=(96, 96))
    frames.append(frame)

f = open(skeleton, "r")
json_data = json.load(f)

for data in json_data['infos']:
    xyz = []
    for p in data['positions']:
        xyz.append(p['x'])
        xyz.append(p['y'])
        xyz.append(p['z'])
    xyzs.append(xyz)

frames = np.array(frames)
xyzs = np.array(xyzs)

sliceIdx = min(len(frames), len(xyzs))
sliceIdx -= (sliceIdx % DEVIDER)
if sliceIdx % DEVIDER:
    frames = frames[:sliceIdx]
    xyzs = xyzs[:sliceIdx]
elif not sliceIdx % DEVIDER:
    frames = frames[:sliceIdx]
    xyzs = xyzs[:sliceIdx]

from tensorflow.keras.applications.vgg16 import VGG16
base_model = VGG16(weights='imagenet', include_top=False)
frames = base_model.predict(frames).reshape(-1, feature_size)

frames, xyzs = frames.reshape(-1, DEVIDER, feature_size), xyzs.reshape(-1, DEVIDER, 96)

result = model.predict([frames, xyzs])
prediction = np.argmax(result, axis=1)
pred = np.unique(prediction, return_counts=True)
sorter = np.argsort(pred[1])
predict = pred[0][sorter][-1]

# 파일명 순으로 정렬
dirIdx = ['1. 진자운동', '10. 90도 외전 후 내회전,외회전', '11. 톱니근 펀치', '12. 동적 베어 허그', '13. 다리직거상 운동', '14. 능동보조 무릎 굴곡 신전(누운)',
          '15. 능동보조 무릎 굴곡 신전(앉은)', '16. 계단 스트레칭', '17. 와위 고관절 외전', '18. 무릎 굴곡 신전(누운)', '19. 무릎 굴곡 신전(앉은)',
          '2. 능동보조 전방굴곡', '20. 햄스트링 강화', '21. 스쿼트', '22. 스텝박스 오르기', '23. 뒤꿈치 슬라이드(누운)', '24. 뒤꿈치 슬라이드(앉은)',
          '25. 뒤꿈치 들기', '26. 빈자전거 운동', '27. 레그프레스', '28. 능동보조 팔꿈치 굴곡 신전', '29. 능동보조 전완부 내회전,외회전', '3. 능동보조 내회전,외회전',
          '30. 능동보조 손목 굴곡 신전', '31. 수부그립', '32. 팔꿈치 굴곡 신전', '33. 전완부 내회전, 외회전', '34. 손목 굴곡 신전', '35. 손목 내전, 외전',
          '36. 오버헤드 팔꿈치 신전', '37. 삼두근 딥스', '38. 능동보조 발목 굴곡 신전', '39. 발목 굴곡 신전', '4. 능동보조 내전', '5. 활차운동',
          '6. 팔꿈치 몸통 붙이기', '7. 능동 전방 굴곡', '8. 능동 내회전, 외회전', '9. 능동 외전']
# print(prediction)
print(dirIdx[predict])  # 예측 결과 도출