import tensorflow as tf
from tensorflow.keras.callbacks import Callback

# Define a custom callback to stop training when accuracy reaches a threshold
class StopAtAccuracy(Callback):
    def __init__(self, target_accuracy):
        super(StopAtAccuracy, self).__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        # Check if the accuracy has reached the target
        accuracy = logs.get("categorical_accuracy")
        if accuracy is not None and accuracy >= self.target_accuracy:
            print(f"\nReached {self.target_accuracy*100}% accuracy, stopping training!")
            self.model.stop_training = True