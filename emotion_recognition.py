import cv2 as cv
from keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('emotion_model.h5')

emotions=["angry","happy","sad","surprise"]

# Load Haar cascade only once
haar_cascade = cv.CascadeClassifier("haar_cascade.xml")

# Open webcam
vid = cv.VideoCapture(0)

while True:
    isTrue, frame = vid.read()

    if not isTrue:
        print("Failed to grab frame.")
        break

    # Detect faces
    face_rect = haar_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=9)

    # Draw rectangles around faces
    for (x, y, w, h) in face_rect:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        face = frame[y:y+h, x:x+w]  # Crop face
        face = cv.resize(face, (48, 48))       # Resize
        face = cv.cvtColor(face, cv.COLOR_BGR2RGB)  # Convert BGR to RGB
        face = face.astype("float32") / 255.0
        face = face.reshape((1, 48, 48, 3))     # Add batch dimension
    
    prediction=model.predict(face)
    class_index = np.argmax(prediction)
    emotion = emotions[class_index]
    cv.putText(frame,emotion, (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)

    # Show the frame with rectangles
    cv.imshow("Detected Faces", frame)

    # Break on pressing 'd'
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

# Release resources
vid.release()
cv.destroyAllWindows()
