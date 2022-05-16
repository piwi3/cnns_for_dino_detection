import sys
import logging
import os
import cv2
from datetime import datetime, timedelta
from utils import predict_frame, key_action, init_cam, predict_frame, add_text #write_image
from tensorflow.keras.models import load_model

# Import pretrained CNN
model_name = 'cnn_pretrained' #'scikeras_cnn'
path = f'/Users/philipwitte/Documents/spiced_projects/fenugreek-student-code/week09/imageclassifier/models/{model_name}'
model = load_model(path)
classes = ['empty', 'triceratops', 'stegosaurus', 'hand_only', 't-rex', 'brachiosaurus']


if __name__ == "__main__":

    # folder to write images to
    # out_folder = sys.argv[1] # To be activated if you want to save images

    # maybe you need this
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    logging.getLogger().setLevel(logging.INFO)
   
    # also try out this resolution: 640 x 360
    webcam = init_cam(640, 480)
    key = None

    # Required global variables
    datetime_prv = datetime.now()
    prediction, prob = ('', 0)

    try:
        # q key not pressed 
        while key != 'q':
            # Capture frame-by-frame
            ret, frame = webcam.read()

            # fliping the image 
            frame = cv2.flip(frame, 1)
   
            # draw a [224x224] rectangle into the frame, leave some space for the black border 
            offset = 2
            width = 224
            x = 160
            y = 120
            color = (0, 0, 0)
            if prediction not in ['', 'empty', 'hand_only']:
                color = (0, 255, 0)
            
            cv2.rectangle(img=frame, 
                          pt1=(x-offset,y-offset), 
                          pt2=(x+width+offset, y+width+offset), 
                          color=color,
                          thickness=2
            )     
            
            # Show prediction on screen
            if datetime.now() >= datetime_prv + timedelta(seconds=1):
                 image = frame[y:y+width, x:x+width, :]
                 prediction, prob = predict_frame(image, model, classes)
                 datetime_prv = datetime.now()

            text = f'Prediction: {prediction} ({round(prob, 1)}%)'
            add_text(text, frame, color)

            # get key event
            key = key_action()
            
            if key == 'space':
                # >> Code for capturing images -> To be uncommented, if you want to save images
                # write the image without overlay
                # extract the [224x224] rectangle out of it
                # image = frame[y:y+width, x:x+width, :]
                # write_image(out_folder, image) 

                # >> Only print text showing prediction result in terminal
                print(text)
            
            # disable ugly toolbar
            cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)              
            
            # display the resulting frame
            cv2.imshow('frame', frame)            
            
    finally:
        # when everything done, release the capture
        logging.info('quit webcam')
        webcam.release()
        cv2.destroyAllWindows()
