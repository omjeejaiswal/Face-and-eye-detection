

from flask import Flask, render_template, Response
import cv2
import os
import time

app = Flask(__name__)
camera = cv2.VideoCapture(0)


SAVE_DIR = "saved_images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def save_image(frame, face_index):
    filename = os.path.join(SAVE_DIR, f"face_{face_index}.jpg")
    
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 7)
    
    for x, y, w, h in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        
    
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        
        if len(eyes) > 0:  
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            for ex, ey, ew, eh in eyes: 
                center = (int(x + ex + ew/2), int(y + ey + eh/2))
                radius = int(min(ew, eh) / 2)
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
            
        
            cv2.putText(frame, 'Face Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.putText(frame, 'Eyes Detected', (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            

            cv2.imwrite(filename, frame)
            print(f"Image saved: {filename}")
            return  
    
    
    print("no face and eye detected... check your camera")

def gen_frames():
    face_index = 0
    save_delay = 0  
    last_save_time = time.time()
    
    while True:

        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            if time.time() - last_save_time >= save_delay:
                save_image(frame, face_index)
                face_index += 1
                last_save_time = time.time()

            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

            frame = print("frame is not working")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)










































































































































































































































































































































































































































































































































































































# from flask import Flask, render_template, Response
# import cv2

# app = Flask(__name__)
# camera = cv2.VideoCapture(0)


# def gen_frames():
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             detector = cv2.CascadeClassifier( cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#             eye_cascade = cv2.CascadeClassifier( cv2.data.haarcascades + "haarcascade_eye.xml")
#             faces = detector.detectMultiScale(frame, 1.1, 7)
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             # Draw the rectangle around each face
#             for x, y, w, h in faces:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#                 roi_gray = gray[y : y + h, x : x + w]
#                 roi_color = frame[y : y + h, x : x + w]

#                 # detect eye within the face
#                 #detectMultiScale(image, rejectLevels, levelWeights)
#                 eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3) 
#                 for ex, ey, ew, eh in eyes:
#                     cv2.rectangle(
#                         roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
#                     )

#             ret, buffer = cv2.imencode(".jpg", frame)
#             frame = buffer.tobytes()
#             yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/video_feed")
# def video_feed():
#     return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# if __name__ == "__main__":
#     app.run(debug=True)














# ---------------- it detect face and eye with mentioned it.  -----------------------------------


# from flask import Flask, render_template, Response
# import cv2

# app = Flask(__name__)
# camera = cv2.VideoCapture(0)

# def gen_frames():
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#             eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
#             faces = detector.detectMultiScale(frame, 1.1, 7)
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             # Draw the rectangle around each face
#             for x, y, w, h in faces:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#                 roi_gray = gray[y:y + h, x:x + w]
#                 roi_color = frame[y:y + h, x:x + w]

#                 # detect eyes within the face
#                 eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
#                 for ex, ey, ew, eh in eyes:

#                     # cv2.rectangle(
#                     #     roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
#                     # )

#                     # Draw circle around each eye
#                     center = (int(ex + ew/2), int(ey + eh/2))
#                     radius = int(min(ew, eh) / 2)
#                     cv2.circle(roi_color, center, radius, (0, 255, 0), 2)

#                 # Add text overlay for face detection
#                 cv2.putText(frame, 'Face Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#                 cv2.putText(frame, 'Eyes Detected', (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
#             ret, buffer = cv2.imencode(".jpg", frame)
#             frame = buffer.tobytes()
#             yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/video_feed")
# def video_feed():
#     return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# if __name__ == "__main__":
#     app.run(debug=True)






















































# ------------------ in this code you can save photos per frame  --------------------------



# from flask import Flask, render_template, Response
# import cv2
# import os

# app = Flask(__name__)
# camera = cv2.VideoCapture(0)

# # Create directory to save images if not exists
# SAVE_DIR = "saved_images"
# if not os.path.exists(SAVE_DIR):
#     os.makedirs(SAVE_DIR)

# def save_image(frame, face_index):
#     filename = os.path.join(SAVE_DIR, f"face_{face_index}.jpg")
#     cv2.imwrite(filename, frame)
#     print(f"Image saved: {filename}")

# def gen_frames():
#     face_index = 0
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#             eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
#             faces = detector.detectMultiScale(frame, 1.1, 7)
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             # Draw the rectangle around each face
#             for x, y, w, h in faces:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#                 roi_gray = gray[y:y + h, x:x + w]
#                 roi_color = frame[y:y + h, x:x + w]

#                 # detect eyes within the face
#                 eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
#                 if len(eyes) > 1:  # If both eyes detected
#                     # Save the image if both eyes are detected
#                     save_image(frame, face_index)
#                     face_index += 1

#                     for ex, ey, ew, eh in eyes:
#                         # Draw circle around each eye
#                         center = (int(ex + ew/2), int(ey + eh/2))
#                         radius = int(min(ew, eh) / 2)
#                         cv2.circle(roi_color, center, radius, (0, 255, 0), 2)

#                     # Add text overlay for face detection
#                     cv2.putText(frame, 'Face Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#                     cv2.putText(frame, 'Eyes Detected', (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#             ret, buffer = cv2.imencode(".jpg", frame)
#             frame = buffer.tobytes()
#             yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/video_feed")
# def video_feed():
#     return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# if __name__ == "__main__":
#     app.run(debug=True)




























# time stamp


# from flask import Flask, render_template, Response
# import cv2
# import os
# import time

# app = Flask(__name__)
# camera = cv2.VideoCapture(0)

# # Create directory to save images if not exists
# SAVE_DIR = "saved_images"
# if not os.path.exists(SAVE_DIR):
#     os.makedirs(SAVE_DIR)

# def save_image(frame, face_index):
#     filename = os.path.join(SAVE_DIR, f"face_{face_index}.jpg")
#     cv2.imwrite(filename, frame)
#     print(f"Image saved: {filename}")

# def gen_frames():
#     face_index = 0
#     save_delay = 2  # Delay between saving consecutive images (in seconds)
#     last_save_time = time.time()
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#             eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
#             faces = detector.detectMultiScale(frame, 1.1, 7)
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             # Draw the rectangle around each face
#             for x, y, w, h in faces:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#                 roi_gray = gray[y:y + h, x:x + w]
#                 roi_color = frame[y:y + h, x:x + w]

#                 # detect eyes within the face
#                 eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
#                 if len(eyes) > 1:  # If both eyes detected
#                     # Save the image if both eyes are detected and delay has passed
#                     if time.time() - last_save_time >= save_delay:
#                         save_image(frame, face_index)
#                         face_index += 1
#                         last_save_time = time.time()

#                     for ex, ey, ew, eh in eyes:
#                         # Draw circle around each eye
#                         center = (int(ex + ew/2), int(ey + eh/2))
#                         radius = int(min(ew, eh) / 2)
#                         cv2.circle(roi_color, center, radius, (0, 255, 0), 2)

#                     # Add text overlay for face detection
#                     cv2.putText(frame, 'Face Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#                     cv2.putText(frame, 'Eyes Detected', (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#             ret, buffer = cv2.imencode(".jpg", frame)
#             frame = buffer.tobytes()
#             yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/video_feed")
# def video_feed():
#     return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# if __name__ == "__main__":
#     app.run(debug=True)



































# unlocked 


# from flask import Flask, render_template, Response
# import cv2

# app = Flask(__name__)
# camera = cv2.VideoCapture(0)

# # Load a pre-trained face recognition model
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # Function to check if a face is detected
# def is_face_detected(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     return len(faces) > 0

# def gen_frames():
#     unlocked = False
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             # Check if a face is detected
#             if is_face_detected(frame):
#                 unlocked = True
#             else:
#                 unlocked = False

#             # Draw "Unlocked" text if face is detected
#             if unlocked:
#                 cv2.putText(frame, "Unlocked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             # Save a photo when 's' key is pressed
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('s'):
#                 cv2.imwrite('live_camera_photo.jpg', frame)
#                 print("Live camera photo saved as 'live_camera_photo.jpg'.")

#             ret, buffer = cv2.imencode(".jpg", frame)
#             frame = buffer.tobytes()
#             yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/video_feed")
# def video_feed():
#     return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# if __name__ == "__main__":
#     app.run(debug=True)





























# just write the face is detected


# from flask import Flask, render_template, Response
# import cv2

# app = Flask(__name__)
# camera = cv2.VideoCapture(0)


# def gen_frames():
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#             eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
#             faces = detector.detectMultiScale(frame, 1.1, 7)
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
#             # Draw the rectangle around each face and add text overlay
#             for x, y, w, h in faces:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#                 cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#                 roi_gray = gray[y:y + h, x:x + w]
#                 roi_color = frame[y:y + h, x:x + w]

#                 # Detect eyes within each face region
#                 eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
#                 for ex, ey, ew, eh in eyes:
#                     cv2.rectangle(
#                         roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
#                     )

#             ret, buffer = cv2.imencode(".jpg", frame)
#             frame = buffer.tobytes()
#             yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/video_feed")
# def video_feed():
#     return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# if __name__ == "__main__":
#     app.run(debug=True)
























# with face_recognition

# import face_recognition
# import cv2
# import numpy as np
# import csv
# import os
# from datetime import datetime
 
# video_capture = cv2.VideoCapture(0)
 
# jobs_image = face_recognition.load_image_file("photos/jobs.jpg")
# jobs_encoding = face_recognition.face_encodings(jobs_image)[0]
 
# ratan_tata_image = face_recognition.load_image_file("photos/tata.jpg")
# ratan_tata_encoding = face_recognition.face_encodings(ratan_tata_image)[0]
 
# sadmona_image = face_recognition.load_image_file("photos/sadmona.jpg")
# sadmona_encoding = face_recognition.face_encodings(sadmona_image)[0]
 
# tesla_image = face_recognition.load_image_file("photos/tesla.jpg")
# tesla_encoding = face_recognition.face_encodings(tesla_image)[0]
 
# known_face_encoding = [
# jobs_encoding,
# ratan_tata_encoding,
# sadmona_encoding,
# tesla_encoding
# ]
 
# known_faces_names = [
# "jobs",
# "ratan tata",
# "sadmona",
# "tesla"
# ]
 
# students = known_faces_names.copy()
 
# face_locations = []
# face_encodings = []
# face_names = []
# s=True
 
 
# now = datetime.now()
# current_date = now.strftime("%Y-%m-%d")
 
 
 
# f = open(current_date+'.csv','w+',newline = '')
# lnwriter = csv.writer(f)
 
# while True:
#     _,frame = video_capture.read()
#     small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
#     rgb_small_frame = small_frame[:,:,::-1]
#     if s:
#         face_locations = face_recognition.face_locations(rgb_small_frame)
#         face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
#         face_names = []
#         for face_encoding in face_encodings:
#             matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
#             name=""
#             face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
#             best_match_index = np.argmin(face_distance)
#             if matches[best_match_index]:
#                 name = known_faces_names[best_match_index]
 
#             face_names.append(name)
#             if name in known_faces_names:
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 bottomLeftCornerOfText = (10,100)
#                 fontScale              = 1.5
#                 fontColor              = (255,0,0)
#                 thickness              = 3
#                 lineType               = 2
 
#                 cv2.putText(frame,name+' Present', 
#                     bottomLeftCornerOfText, 
#                     font, 
#                     fontScale,
#                     fontColor,
#                     thickness,
#                     lineType)
 
#                 if name in students:
#                     students.remove(name)
#                     print(students)
#                     current_time = now.strftime("%H-%M-%S")
#                     lnwriter.writerow([name,current_time])
#     cv2.imshow("attendence system",frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
 
# video_capture.release()
# cv2.destroyAllWindows()
# f.close()

# yt channel - https://www.youtube.com/watch?v=A6464U4bPPQ
# code link -- https://i-know-python.com/facial-recognition-attendance-system-using-python/



































# without importing face_recogization
# and using flask


# from flask import Flask, render_template, Response
# import cv2
# import numpy as np
# import csv
# from datetime import datetime

# app = Flask(__name__)
# camera = cv2.VideoCapture(0)

# # Load pre-trained face detection model
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Load images and their names
# jobs_image = cv2.imread("photos/jobs.jpg")
# ratan_tata_image = cv2.imread("photos/tata.jpg")
# sadmona_image = cv2.imread("photos/sadmona.jpg")
# tesla_image = cv2.imread("photos/tesla.jpg")

# # Convert images to grayscale
# jobs_gray = cv2.cvtColor(jobs_image, cv2.COLOR_BGR2GRAY)
# ratan_tata_gray = cv2.cvtColor(ratan_tata_image, cv2.COLOR_BGR2GRAY)
# sadmona_gray = cv2.cvtColor(sadmona_image, cv2.COLOR_BGR2GRAY)
# tesla_gray = cv2.cvtColor(tesla_image, cv2.COLOR_BGR2GRAY)

# # Detect faces in the images
# jobs_faces = face_cascade.detectMultiScale(jobs_gray, scaleFactor=1.1, minNeighbors=5)
# ratan_tata_faces = face_cascade.detectMultiScale(ratan_tata_gray, scaleFactor=1.1, minNeighbors=5)
# sadmona_faces = face_cascade.detectMultiScale(sadmona_gray, scaleFactor=1.1, minNeighbors=5)
# tesla_faces = face_cascade.detectMultiScale(tesla_gray, scaleFactor=1.1, minNeighbors=5)

# # Extract face encodings (not implemented in OpenCV, but can be done with more advanced methods)

# known_faces = {
#     "jobs": jobs_faces,
#     "ratan tata": ratan_tata_faces,
#     "sadmona": sadmona_faces,
#     "tesla": tesla_faces
# }

# students = list(known_faces.keys())

# def recognize_face(frame):
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
#     for (x, y, w, h) in faces:
#         # Perform face recognition (not implemented in OpenCV)
#         # Here you would compare the detected face with known faces using a recognition algorithm
        
#         # For simplicity, let's just draw rectangles around the detected faces
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
#         # Placeholder for recognizing the face and updating attendance
        
#     return frame

# def gen_frames():
#     now = datetime.now()
#     current_date = now.strftime("%Y-%m-%d")
#     f = open(current_date+'.csv','w+',newline='')
#     lnwriter = csv.writer(f)
    
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
        
#         # Perform face recognition
#         frame = recognize_face(frame)
        
#         ret, buffer = cv2.imencode(".jpg", frame)
#         frame = buffer.tobytes()
#         yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

#     f.close()

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/video_feed")
# def video_feed():
#     return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# if __name__ == "__main__":
#     app.run(debug=True)






















