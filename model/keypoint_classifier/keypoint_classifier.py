#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        # Get the raw output tensor from the TFLite model
        result = self.interpreter.get_tensor(output_details_tensor_index)

        # Squeeze the array to get a 1D list of probabilities
        squeeze_result = np.squeeze(result)

        # Find the index of the highest probability (the predicted sign)
        result_index = np.argmax(squeeze_result)
        
        # Get the actual confidence score (percentage) at that index
        confidence = float(squeeze_result[result_index])

        # Return both the class index and the confidence score
        return result_index, confidence