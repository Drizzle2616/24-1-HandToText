# 필요한 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import cv2
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
from tkinter import messagebox

# 1. 데이터 수집 및 탐색
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 데이터 탐색
plt.figure(figsize=(5, 5))
plt.imshow(train_images[0], cmap='gray')
plt.title(f'Label: {train_labels[0]}')
plt.show()

# 2. 데이터 전처리
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# 3. 모델 구축 및 학습
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'CNN Accuracy: {test_acc}')

# 4. 예측 및 평가
# 혼동 행렬 시각화 (CNN)
cnn_predictions = np.argmax(model.predict(test_images), axis=-1)
cm = tf.math.confusion_matrix(np.argmax(test_labels, axis=1), cnn_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - CNN')
plt.show()

# 이미지 전처리 함수
def load_and_preprocess_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    image = image / 255.0
    image = image.reshape(28, 28, 1)
    return image

# 새로운 이미지 예측 예제
file_path = 'path_to_your_new_image.png'  # 손글씨 이미지 파일 경로
new_image = load_and_preprocess_image(file_path)
new_image = np.expand_dims(new_image, axis=0)
prediction = model.predict(new_image)
predicted_label = np.argmax(prediction)
print(f'Predicted Label: {predicted_label}')

# 5. 사용자 인터페이스 (GUI)
class HandwritingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Handwriting Recognition')
        self.canvas = tk.Canvas(self, width=280, height=280, bg='white')
        self.canvas.pack()
        self.button_predict = tk.Button(self, text='Predict', command=self.predict)
        self.button_predict.pack()
        self.button_clear = tk.Button(self, text='Clear', command=self.clear)
        self.button_clear.pack()
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind('<B1-Motion>', self.draw_lines)

    def draw_lines(self, event):
        x, y = event.x, event.y
        self.canvas.create_line(x, y, x+1, y+1, fill='black', width=8)
        self.draw.line([x, y, x+1, y+1], fill='black', width=8)

    def clear(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        self.image = self.image.resize((28, 28))
        self.image = ImageOps.invert(self.image)
        image_array = np.array(self.image) / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)
        prediction = model.predict(image_array)
        predicted_label = np.argmax(prediction)
        messagebox.showinfo('Prediction', f'Predicted Label: {predicted_label}')

if __name__ == '__main__':
    app = HandwritingApp()
    app.mainloop()
