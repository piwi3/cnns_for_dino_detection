import logging
import os
from datetime import datetime
import cv2
import numpy as np


def write_image(out, frame):
    """
    writes frame from the webcam as png file to disk. datetime is used as filename.
    """
    if not os.path.exists(out):
        os.makedirs(out)
    now = datetime.now() 
    dt_string = now.strftime("%H-%M-%S-%f") 
    filename = f'{out}/{dt_string}.png'
    logging.info(f'write image {filename}')
    cv2.imwrite(filename, frame)


def key_action():
    # https://www.ascii-code.com/
    k = cv2.waitKey(1)
    
    if k == 113: # q button
        return 'q'
    if k == 32: # space bar
        return 'space'
    if k == 112: # p key
        return 'p'
    return None


def init_cam(width, height):
    """
    setups and creates a connection to the webcam
    """

    logging.info('start web cam')
    cap = cv2.VideoCapture(1)
    
    # Check success
    if not cap.isOpened():
        raise ConnectionError("Could not open video device")
    
    # Set properties. Each returns === True on success (i.e. correct resolution)
    assert cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    assert cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def add_text(text, frame, color):
    # Define font for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get boundary of text
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    
    # Get coordinates for start/ending of text (pass as integers)
    textX = (frame.shape[1] - textsize[0]) / 2
    textX = int(textX)

    cv2.putText(frame, 
                text, 
                (textX, 80), 
                font, 1, 
                color, 
                2)#, 
               # cv2.LINE_4)
    

def predict_frame(image, model, classes):
   # reverse color channels
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

   # reshape image to (1, 224, 224, 3)
   image = image.reshape(1, 224, 224, 3)

   # apply pre-processing
   prediction = model.predict(image)
   prob = 100 * prediction.max()
   prediction = classes[np.argmax(prediction)]

   # Return text
   return prediction, prob

