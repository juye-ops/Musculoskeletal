from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, concatenate, Dropout
import numpy as np

xf_train = np.load("dataset/xf_train.npy")
yf_train = np.load("dataset/yf_train.npy")
xf_val = np.load("dataset/xf_val.npy")
yf_val = np.load("dataset/yf_val.npy")

xj_train = np.load("dataset/xj_train.npy")
yj_train = np.load("dataset/yj_train.npy")
xj_val = np.load("dataset/xj_val.npy")
yj_val = np.load("dataset/yj_val.npy")

input1 = keras.Input(shape=xf_train.shape[1:3])  # 이미지 레이어
input2 = keras.Input(shape=xj_train.shape[1:3])  # 스켈레톤 레이어

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

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
model.summary()


from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# 사용법 참고: train값) in: [train1, train2] // out: train_y
# 사용법 참고: val값) ([val1, val2], val_y) 형태

history = model.fit(
    [xf_train, xj_train],
    yf_train,
    validation_data=([xf_val, xj_val], yf_val),
    epochs=30,
    callbacks=[
               ModelCheckpoint('weights/model.h5',monitor='val_acc',verbose=1,save_best_only=True,mode='auto'),
               ReduceLROnPlateau(monitor='val_acc', factor=0.5,patience=50,verbose=1,mode='auto')
    ]
)

import matplotlib.pyplot as plt
history.history.keys()

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
