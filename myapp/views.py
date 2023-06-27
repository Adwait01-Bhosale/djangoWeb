from django.shortcuts import render
import cv2
import threading
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import mediapipe as mp
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
from django.http import HttpResponse
import json
from io import BytesIO
from PIL import Image

def homePage(request):
    return render(request, 'base.html')


@csrf_exempt
def videoCam(request):
    if request.method == 'GET':
        print("Inside POST")
        # Retrieve the video data from the request
        video_data = request.POST.get('videoData')

        # Remove the data URL prefix and decode the base64 data
        # video_data = video_data.split(',')[1]
        video_data = base64.b64decode(video_data)
        
        # Convert the video data to a numpy array
        nparr = np.frombuffer(video_data, np.uint8)
        video_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert video to grayscale
        gray_video = cv2.cvtColor(video_array, cv2.COLOR_BGR2GRAY)

        # Encode the processed video data as base64
        _, processed_video_data = cv2.imencode('.jpg', gray_video)
        processed_video_data = processed_video_data.tobytes()
        processed_video_data = base64.b64encode(processed_video_data).decode('utf-8')

        # Return the processed video data as a JSON response
        response = {
            'processedVideoData': processed_video_data,
        }
        return JsonResponse(response)

    # If the request is not a POST, return an error response
    response = {
        'message': 'Invalid request',
    }
    return JsonResponse(response, status=400)




def grayscale(request):
  # Get the base64 string from the request
   pass


@gzip.gzip_page
def kyc(request):
    try:
        cam=VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass
    return render(request, 'home.html')


class VideoCamera(object):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame)=self.video.read()
        threading.Thread(target=self.update, args=()).start()
    
    def __del__(self):
        self.video.release()
        
    
    def get_frame(self):
        image = self.frame
        image = cv2.flip(image, 1)
        
        red = image[:, :, 2].copy()  # Extract red channel
        blue = image[:, :, 0].copy()  # Extract blue channel

        image[:, :, 0] = red  # Replace blue channel with red
        image[:, :, 2] = blue
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        
        # Get the result
        results = self.face_mesh.process(image)
        
        # To improve performance
        image.flags.writeable = True
        
        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                
                if y < -10:
                    left=True
                    text = "Looking Left"
                elif y > 10:
                    right = True
                    text = "Looking Right"
                elif x < -10:
                    down=True
                    text = "Looking Down"
                else:
                    forward=True
                    text = "Forward"
                
                    

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                
                # cv2.line(image, p1, p2, (255, 0, 0), 2)

                # Add the text on the image
                cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Head Pose Estimation', image)

        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    
    def update(self):
        while True:
            (self.grabbed, self.frame)=self.video.read()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')