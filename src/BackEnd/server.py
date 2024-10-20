import os
import cv2 as cv
import csv
import copy
import json
import itertools
from collections import Counter
from collections import deque
import numpy as np
import argparse
import mediapipe as mp
import base64
from utils.cvfpscalc import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
from dotenv import load_dotenv
from flask import Flask, Response, render_template, request
from flask_socketio import SocketIO, emit, send
import utils.functions as uf
import time

# Configuration
load_dotenv()

# Initialization
app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="*")

firstFrame = True

@app.route("/")
def index():
    return render_template('index.html')  # Render an HTML template

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('message')
def handle_message(msg):
    print(f'Received message: {msg}')
    # if msg.lower() == "quit":
    #     camera.release()  # Release the webcam
    #     cv2.destroyAllWindows()  # Close the window
    # if msg.lower() == "start":
    #     if camera.isOpened():
    #         pass
    #     else:
    #         camera = cv2.VideoCapture(1)
    # socketio.send('Good morning!')
# def stream():
#     return Response(recognize(), mimetype="multipart/x-mixed-replace; boundary=frame")
@socketio.on("disconnect")
def on_disconnect():
    client_id = request.sid
    print(f'Client disconnected with SID: {client_id}')
    
def generate_frames():
    # camera = cv.VideoCapture(1)  # Change index if needed
    # while True:
    #     success, frame = camera.read()
    #     if not success:
    #         break
    #     else:
    #         ret, buffer = cv.imencode('.jpg', frame)
    #         frame = buffer.tobytes()
    #         yield (b'--frame\r\n'
    #                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    # Argument parsing #################################################################
    args = uf.get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    camera = cv.VideoCapture(cap_device)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    if camera.isOpened():
        print("Webcam is ready and opened successfully.")
        # ret, image = camera.read()
        # cv.imshow('Hand Gesture Recognition', image)
    else:
        print("Webcam is not ready.")

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = uf.select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = camera.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = uf.calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = uf.calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = uf.pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = uf.pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                uf.logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                debug_image = uf.draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = uf.draw_landmarks(debug_image, landmark_list)
                debug_image = uf.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
                # print('Text: ' + keypoint_classifier_labels[hand_sign_id])
                # print('Gesture Text: ' + point_history_classifier_labels[most_common_fg_id[0][0]])
                socketio.send(keypoint_classifier_labels[hand_sign_id])
        else:
            point_history.append([0, 0])

        debug_image = uf.draw_point_history(debug_image, point_history)
        debug_image = uf.draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        # cv.imshow('Hand Gesture Recognition', debug_image)
        ret, buffer = cv.imencode('.jpg', debug_image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()
    cv.destroyAllWindows()
# @socketio.on('video-frame')
# def handle_video_frame(data):
#     # Decode the Base64-encoded frame data
#     # client_id = request.sid
#     # print(f'Data from {client_id}')
#     global firstFrame
#     # hands: mp.python.solutions.hands.Hands
#     mode = 0
#     use_brect: bool
#     keypoint_classifier: KeyPointClassifier
#     point_history_classifier: PointHistoryClassifier
#     keypoint_classifier_labels: list[str]
#     point_history_classifier_labels: list[str]
#     history_length: int
#     point_history: deque
#     finger_gesture_history: deque
#     # Initialize MediaPipe Hands
#     mp_hands = mp.solutions.hands
#     # Create an instance of Hands with default configurations
#     hands = mp_hands.Hands()
#     if firstFrame :
#         firstFrame = False
#         args = uf.get_args()

#         # cap_device = args.device
#         # cap_width = args.width
#         # cap_height = args.height

#         use_static_image_mode = args.use_static_image_mode
#         min_detection_confidence = args.min_detection_confidence
#         min_tracking_confidence = args.min_tracking_confidence

#         use_brect = True
#         # Model load #############################################################
#         mp_hands = mp.solutions.hands
#         hands = mp_hands.Hands(
#             static_image_mode=use_static_image_mode,
#             max_num_hands=1,
#             min_detection_confidence=min_detection_confidence,
#             min_tracking_confidence=min_tracking_confidence,
#         )

#         keypoint_classifier = KeyPointClassifier()

#         point_history_classifier = PointHistoryClassifier()

#         # Read labels ###########################################################
#         with open('model/keypoint_classifier/keypoint_classifier_label.csv',
#                 encoding='utf-8-sig') as f:
#             keypoint_classifier_labels = csv.reader(f)
#             keypoint_classifier_labels = [
#                 row[0] for row in keypoint_classifier_labels
#             ]
#         with open(
#                 'model/point_history_classifier/point_history_classifier_label.csv',
#                 encoding='utf-8-sig') as f:
#             point_history_classifier_labels = csv.reader(f)
#             point_history_classifier_labels = [
#                 row[0] for row in point_history_classifier_labels
#             ]
#         # FPS Measurement ########################################################
#         cvFpsCalc = CvFpsCalc(buffer_len=10)

#         # Coordinate history #################################################################
#         history_length = 16
#         point_history = deque(maxlen=history_length)

#         # Finger gesture history ################################################
#         finger_gesture_history = deque(maxlen=history_length)

#         #  ########################################################################
#         print('First thing first')
#     else:
#         image_in = base64_to_image(data)
#         if image_in is None:
#             pass
#         else:   
#             # print('processing...')
#             image = cv.flip(image_in, 1)  # Mirror display
#             # fps = cvFpsCalc.get()
#             # Process Key (ESC: end) #################################################
#             key = cv.waitKey(10)
            
#             number, mode = uf.select_mode(key, mode)
#             debug_image = copy.deepcopy(image)

#             # Detection implementation #############################################################
#             image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

#             image.flags.writeable = False
#             results = hands.process(image)
#             image.flags.writeable = True

#                 #  ####################################################################
#             if results.multi_hand_landmarks is not None:
#                 for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
#                                                         results.multi_handedness):
#                         # Bounding box calculation
#                         brect = uf.calc_bounding_rect(debug_image, hand_landmarks)
#                         # Landmark calculation
#                         landmark_list = uf.calc_landmark_list(debug_image, hand_landmarks)

#                         # Conversion to relative coordinates / normalized coordinates
#                         pre_processed_landmark_list = uf.pre_process_landmark(
#                             landmark_list)
#                         pre_processed_point_history_list = uf.pre_process_point_history(
#                             debug_image, point_history)
#                         # Write to the dataset file
#                         uf.logging_csv(number, mode, pre_processed_landmark_list,
#                                     pre_processed_point_history_list)

#                         # Hand sign classification
#                         hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
#                         if hand_sign_id == 2:  # Point gesture
#                             point_history.append(landmark_list[8])
#                         else:
#                             point_history.append([0, 0])

#                         # Finger gesture classification
#                         finger_gesture_id = 0
#                         point_history_len = len(pre_processed_point_history_list)
#                         if point_history_len == (history_length * 2):
#                             finger_gesture_id = point_history_classifier(
#                                 pre_processed_point_history_list)

#                         # Calculates the gesture IDs in the latest detection
#                         finger_gesture_history.append(finger_gesture_id)
#                         most_common_fg_id = Counter(
#                             finger_gesture_history).most_common()

#                         # Drawing part
#                         debug_image = uf.draw_bounding_rect(use_brect, debug_image, brect)
#                         debug_image = uf.draw_landmarks(debug_image, landmark_list)
#                         debug_image = uf.draw_info_text(
#                             debug_image,
#                             brect,
#                             handedness,
#                             keypoint_classifier_labels[hand_sign_id],
#                             point_history_classifier_labels[most_common_fg_id[0][0]],
#                         )
#                         # print('Text: ' + keypoint_classifier_labels[hand_sign_id])
#                         # print('Gesture Text: ' + point_history_classifier_labels[most_common_fg_id[0][0]])
#                         socketio.send(keypoint_classifier_labels[hand_sign_id])
#                 else:
#                     point_history.append([0, 0])
    
    
#     # yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#     # Process the frame as needed
#     # ...
#     # # Example processing (convert to grayscale)
#     # gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
#     # # Encode processed image back to base64
#     # _, buffer = cv.imencode('.jpg', gray_image)
#     # encoded_image = base64.b64encode(buffer).decode('utf-8')
#     # socketio.emit('display-frame', f"data:image/jpeg;base64,{encoded_image}")
    
# def base64_to_image(base64_string):
#     try:
#         # Extract the base64 encoded binary data from the input string
#         base64_data = base64_string.split(",")[1]
#         # Decode the base64 data to bytes
#         image_bytes = base64.b64decode(base64_data)
#         # Convert the bytes to numpy array
#         image_array = np.frombuffer(image_bytes, dtype=np.uint8)
#         # Decode the numpy array as an image using OpenCV
#         image = cv.imdecode(image_array, cv.IMREAD_COLOR)
#         return image  
#     except:
#         return None
#     # filename = os.path.join(f'image_{int(time.time())}.jpg')
#     # Save the image
#     # cv.imwrite(filename, image)
@app.route('/stream')
def stream():
    return render_template('stream.html')
if __name__ == "__main__":
    # # Get the OpenCV version
    # version = cv2.__version__

    # # Print the version
    # print("OpenCV version:", version)
    # print("Server is listening on port 1234")
    # if camera.isOpened():
    #     print("Webcam is ready and opened successfully.")
    #     ret, image = camera.read()
    #     cv2.imshow('Hand Gesture Recognition', image)
    # else:
    #     print("Webcam is not ready.")
    socketio.run(app, debug=False, port=1234)
    

