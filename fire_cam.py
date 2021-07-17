import numpy as np
import cv2
import tensorflow as tf
import os
import time
def predict_now(img ,model):


    resize = cv2.resize(img ,(160 ,160))
    rgb = cv2.cvtColor(resize ,cv2.COLOR_BGR2RGB)
    test2 = rgb[np.newaxis is None ,: ,: ,:]


    start = time.time()
    predictions = model.predict(test2)
    end = time.time()
    # os.system('clear')

    print('Predictions:{} Time taken: {}\n'.format(predictions[0][0], end -start))


    if predictions < -1 :
        print('Fire hazard!\n')


    elif predictions > -1 and predictions < 1.5:
        print('Warning posibility of fire\n')



    else:
        print('No fire hazard\n')

    time.sleep(1)
    print('New image capture')

    return predictions[0][0]



BASE_DIR = os.path.dirname(os.path.abspath('fire_detection'))
path_to_model = os.path.join(BASE_DIR, 'fire_detect_model')



model = tf.keras.models.load_model('fire_detect_model')









def video():
    if __name__ == '__main__':


        cap = cv2.VideoCapture(0)
        start = time.time()
        diff =1
        while(1):
            ret, img = cap.read();

            if int(diff ) %6 == 0:
                predict_now(img ,model)



            cv2.imshow('image', img)


            end = time.time()
            diff = end -start

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
video()
