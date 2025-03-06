import cv2
from deepface import DeepFace

# 載入圖片
img = cv2.imread('girl.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 眼睛模型
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
eyes = eye_cascade.detectMultiScale(gray)
for (x, y, w, h) in eyes:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 標記綠色方框

# 嘴巴模型
mouth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
mouths = mouth_cascade.detectMultiScale(gray)
for (x, y, w, h) in mouths:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)  # 標記紅色方框

# 鼻子模型
nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
noses = nose_cascade.detectMultiScale(gray)
for (x, y, w, h) in noses:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  # 標記藍色方框

# 偵測臉部並進行情緒識別
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray)

for (x, y, w, h) in faces:
    # 取得人臉區域
    face_img = img[y:y+h, x:x+w]
    
    # 使用 DeepFace 進行情緒識別
    result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)

    # 顯示情緒
    emotion = result[0]['dominant_emotion']
    cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # 標記人臉
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)  # 輸出黃色方框

# 顯示結果
cv2.imshow('Emotion Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()