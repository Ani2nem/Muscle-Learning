import cv2
import numpy as np
from sklearn.svm import LinearSVC
from PIL import Image


class Model:

    def __init__(self):
        self.model = LinearSVC(max_iter=10000)

    def train_model(self, counters):
        img_list = []
        class_list = []

        for i in range(1, counters[0]):
            img = cv2.imread(f"1/frame{i}.jpg", cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (150, 150))
            img = img.flatten()
            img_list.append(img)
            class_list.append(1)

        for i in range(1, counters[1]):
            img = cv2.imread(f"2/frame{i}.jpg", cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (150, 150))
            img = img.flatten()
            img_list.append(img)
            class_list.append(2)

        img_list = np.array(img_list)
        class_list = np.array(class_list)

        self.model.fit(img_list, class_list)
        print("Model successfully trained!")

    def predict(self, frame):
        frame = frame[1]
        cv2.imwrite("frame.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        img = Image.open("frame.jpg")
        img = img.resize((150, 150))
        img.save("frame.jpg")

        img = cv2.imread("frame.jpg", cv2.IMREAD_GRAYSCALE)
        img = img.flatten()
        prediction = self.model.predict([img])
        return prediction[0]
