from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Create a VideoCapture object to access the camera
cap = cv2.VideoCapture(0)

painting = cv2.imread('painting1.jpg')
if painting is None:
    raise Exception("Could not load the painting image. Make sure the file 'painting1.jpg' exists.")

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            painting_resized = cv2.resize(painting, (w, h))
            mask_resized = cv2.resize(mask[y:y+h, x:x+w], (w, h))
            frame[y:y+h, x:x+w] = cv2.bitwise_and(frame[y:y+h, x:x+w], frame[y:y+h, x:x+w], mask=~mask_resized)
            frame[y:y+h, x:x+w] += cv2.bitwise_and(painting_resized, painting_resized, mask=mask_resized)


        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5002)