import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
import utils.MediaPipeUtilities as mpfs
from dotenv import load_dotenv
from datetime import datetime
from utils.StopAtAccuracy import StopAtAccuracy
import random


def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def CollectData():
    print('Collecting Data')
    load_dotenv()
    # Path for exported data, numpy arrays
    data_path = os.path.join(os.getenv('DATA_PATH')) 
    no_sequences = os.getenv('no_sequences')
    sequence_length = os.getenv('sequence_length')

    # Actions that we try to detect
    actions = np.array(['hello', 'seeyoulater', 'I', 'father', 'mother', 'yes', 'no', 'help', 'please', 'thankyou', 'want', 'what', 'iloveyou'])
    # actions = np.array(['hello', 'seeyoulater', 'I', 'father', 'mother', 'yes', 'no', 'help'])
    # actions = np.array(['please'])
    for action in actions: 
        for sequence in range(int(no_sequences)):
            try: 
                os.makedirs(os.path.join(data_path, action, str(sequence)))
            except:
                pass
    cap = cv2.VideoCapture(1)
    # Create a named window that allows resizing
    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

    # Resize the window to the desired dimensions (width, height)
    cv2.resizeWindow("Camera Feed", 1024, 768)
    # Set mediapipe model 
    with mpfs.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
        # NEW LOOP
        # Loop through actions
        for action in actions:
            # Loop through sequences aka videos
            for sequence in range(int(no_sequences)):
                # Loop through video length aka sequence length
                for frame_num in range(int(sequence_length)):

                    # Read feed
                    ret, frame = cap.read()
                    corrected_frame = cv2.flip(frame, 1)
                    # Make detections
                    image, results = mpfs.mediapipe_detection(corrected_frame, holistic)
                    # Draw landmarks
                    mpfs.draw_styled_landmarks(image, results)
                    # cv2.imshow('Camera Feed', image)
                    # NEW Apply wait logic
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('Camera Feed', image)
                        cv2.waitKey(2000)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('Camera Feed', image)
                    
                    # NEW Export keypoints
                    keypoints = mpfs.extract_keypoints(results)
                    npy_path = os.path.join(data_path, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
        cap.release()
        cv2.destroyAllWindows()

def Train():
    print('Training----------------')
    load_dotenv()
    # Path for exported data, numpy arrays
    data_path = os.path.join(os.getenv('DATA_PATH')) 
    no_sequences = int(os.getenv('no_sequences'))
    sequence_length = int(os.getenv('sequence_length'))
    actions = np.array(['hello', 'seeyoulater', 'I', 'father', 'mother', 'yes', 'no', 'help', 'please', 'thankyou', 'want', 'what', 'iloveyou'])
    # random.shuffle(actions)
    # actions = np.array(['hello','thankyou', 'want', 'what'])
    label_map = {label:num for num, label in enumerate(actions)}
    sequences, labels = [], []
    
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(data_path, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
      
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    checkpoint_callback = ModelCheckpoint(filepath=f'best_model_{get_timestamp()}.h5', save_best_only=True, monitor='categorical_accuracy', mode='max', verbose=1)
    stop_at_accuracy = StopAtAccuracy(target_accuracy=0.95)
    reduce_lr = ReduceLROnPlateau(monitor='categorical_accuracy', factor=0.2, patience=5, min_lr=0.0001)
    
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))

    # Dropout for regularization
    model.add(Dropout(0.3))

    # Larger Dense layers for handling more complex patterns
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))  # output classes
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
       
    model.fit(X_train, y_train, epochs=4000, callbacks=[tb_callback, checkpoint_callback, stop_at_accuracy, reduce_lr])
    print('Done---------------------------')
    print('Test---------------------------')
    yhat = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    print('Predict:', yhat)
    print('True:', ytrue)
    
def Real_Time():
    print('ReaL time detecting---------------------------')
    # 1. New detection variables
    sequence = []
    threshold = 0.8
    actions = np.array(['hello', 'seeyoulater', 'I', 'father', 'mother', 'yes', 'no', 'help', 'please', 'thankyou', 'want', 'what', 'iloveyou'])
    model = load_model('best_model_20241118_102155.h5')
    cap = cv2.VideoCapture(1)
    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
    # Resize the window to the desired dimensions (width, height)
    cv2.resizeWindow("Camera Feed", 1024, 768)
    # Set mediapipe model 
    with mpfs.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            for num in range(30):
                # Read feed
                ret, frame = cap.read()
                corrected_frame = cv2.flip(frame, 1)
                
                if ret:
                    # Make detections
                    image, results = mpfs.mediapipe_detection(corrected_frame, holistic)
                            
                    # Draw landmarks
                    mpfs.draw_styled_landmarks(image, results)
                    if num == 0: 
                            cv2.putText(image, 'Start Signing', (120,200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('Camera Feed', image)
                            cv2.waitKey(2000)
                    else: 
                            cv2.putText(image, 'Interpreting...', (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('Camera Feed', image)
                    # 2. Prediction logic
                    keypoints = mpfs.extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]
                    
                    if len(sequence) == 30:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        print(res)
                        print(np.argmax(res))
                        if res[np.argmax(res)] > threshold: 
                            print(actions[np.argmax(res)])
                            # cap.release()
                            # cv2.destroyAllWindows()
                # break      
                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
       
def main():
    # pass
    Real_Time()
    # CollectData()   
    # Train() 

if __name__ == "__main__":
    main()