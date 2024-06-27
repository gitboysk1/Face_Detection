import cv2
import numpy as np
from keras.models import model_from_json


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


json_file = open(r'C:\Users\suraj\Desktop\opencv-course-master\Emotion_Detection_using_deep_learning-main\Model\emotion_model.json')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)


emotion_model.load_weights(r"C:\Users\suraj\Desktop\opencv-course-master\Emotion_Detection_using_deep_learning-main\Model\emotion_model.h5")
print("Loaded model from disk")


# cap = cv2.VideoCapture(0)

video_path = 'C:\\Users\\suraj\\Desktop\\Emotion_Detection_using_deep_learning-main\\Data\\Example of Human Facial Expressions _ Emotions (1).mp4'
cap = cv2.VideoCapture(video_path)

# # image_path = 'C:\\Users\\suraj\\Desktop\\emotion_image(2).png'
# image_path = (r"C:\Users\suraj\Desktop\fearful.jpeg")
# frame = cv2.imread(image_path)



while True:
    
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 1020))
    if not ret:
        break
face_detector = cv2.CascadeClassifier('C:\\Program Files\\Python311\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


num_faces = face_detector.detectMultiScale(
    gray_frame, scaleFactor=1.3, minNeighbors=5)


for (x, y, w, h) in num_faces:
    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
    roi_gray_frame = gray_frame[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(
        cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

    
    emotion_prediction = emotion_model.predict(cropped_img)
    maxindex = int(np.argmax(emotion_prediction))
    cv2.putText(frame, emotion_dict[maxindex], (x+5, y+50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from keras.models import model_from_json
# import matplotlib.pyplot as plt

# # Define the emotion dictionary
# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
#                 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# # Load the emotion detection model
# json_file = open('Model\\emotion_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# emotion_model = model_from_json(loaded_model_json)
# emotion_model.load_weights("Model\\emotion_model.h5")
# print("Loaded model from disk")

# # Load the image
# image_path = r"C:\Users\suraj\Desktop\Emotion_Detection_using_deep_learning-main\Data\test\angry\PrivateTest_1623042.jpg"
# frame = cv2.imread(image_path)

# # Perform face detection
# face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

# # Process each face
# for (x, y, w, h) in num_faces:
#     # Extract the face region
#     roi_gray_frame = gray_frame[y:y + h, x:x + w]
#     cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

#     # Perform emotion prediction
#     emotion_prediction = emotion_model.predict(cropped_img)
#     maxindex = int(np.argmax(emotion_prediction))
    
#     # Print the emotion label
#     emotion_label = emotion_dict[maxindex]
#     print(f"Emotion detected: {emotion_label}")

# # Display the image using matplotlib
# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# plt.axis('off')  # Hide axis
# plt.show()
