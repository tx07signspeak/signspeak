import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import utils.MediaPipeUtilities as mpfs
from dotenv import load_dotenv

colors = [(245,117,16), (117,245,16), (16,117,245)]
DATA_PATH = ''
actions = []
no_sequences = 30
sequence_length = 30

def CollectData():
    load_dotenv()
    # Path for exported data, numpy arrays
    DATA_PATH = os.path.join(os.getenv('DATA_PATH')) 

    # Actions that we try to detect
    actions = np.array(['I', 'You', 'Help'])

    # Thirty videos worth of data
    no_sequences = 30

    # Videos are going to be 30 frames in length
    sequence_length = 30

    for action in actions: 
        for sequence in range(no_sequences):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass
    cap = cv2.VideoCapture(1)
    # Set mediapipe model 
    with mpfs.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
        # NEW LOOP
        # Loop through actions
        for action in actions:
            # Loop through sequences aka videos
            for sequence in range(no_sequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):

                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mpfs.mp.mediapipe_detection(frame, holistic)
    #                 print(results)

                    # Draw landmarks
                    mpfs.draw_styled_landmarks(image, results)
                    
                    # NEW Apply wait logic
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                    
                    # NEW Export keypoints
                    keypoints = mpfs.extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                        
        cap.release()
        cv2.destroyAllWindows()
        


def Train():
    load_dotenv()
    actions = np.array(['I', 'You', 'help'])
    label_map = {label:num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    model.load_weights('action.h5')
    model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
    
def main():
    pass
    # CollectData()    

if __name__ == "__main__":
    main()