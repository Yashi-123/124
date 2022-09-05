# import the opencv library
from http.client import _DataType
from locale import normalize
import cv2
import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('keras_model.h5')

  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
    img=cv2.resize(frame,(224,224))
    text=np.array(img,_dType=np.float32)
    text=np.expand_dims(text,axis=0)
    normalized=text/255
    prediction=model.predict(normalized)
    print(prediction)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()