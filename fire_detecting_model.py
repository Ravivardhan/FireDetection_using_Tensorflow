import numpy as np
import cv2 
import tensorflow as tf
import os
import time
import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename





def predict_now(img,model):


    resize = cv2.resize(img,(160,160))
    rgb = cv2.cvtColor(resize,cv2.COLOR_BGR2RGB)
    test2 = rgb[np.newaxis is None,:,:,:]


    start = time.time()
    predictions = model.predict(test2)
    end = time.time()
    #os.system('clear')

    print('Predictions:{} Time taken: {}\n'.format(predictions[0][0], end-start))
    

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
            diff=1
            while(1):
                ret, img = cap.read();

                if int(diff)%6 == 0:
                    predict_now(img,model)



                cv2.imshow('image', img)


                end = time.time()
                diff = end-start

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

def picture(img=''):
    if len(img)==0:
        img=askopenfilename()

    if __name__ == '__main__':

            cap = cv2.imread('{}'.format(img))
            start = time.time()
            diff = 1
            for i in range(1):
                # ret, img = cap.read();

                predict_now(cap, model)

                cv2.imshow('image', cap)

                end = time.time()
                diff = end - start

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()



window=tk.Tk()
window.maxsize(500,500)
window.minsize(500,500)

def get_input():
    img="{}".format(path.get())

    picture(img)



title=Label(window,text="Fire Recognition Using PYTHON",font=("Helvatica",22)).place(x=45,y=50)

path=Entry(window,width=40,text="enter the destination path")
path.place(x=60,y=300)
send=Button(window,text="submit",command=get_input).place(x=400,y=300)

find=Button(window,text="select picture",command=picture).place(x=130,y=330)
web_cam=Button(window,text="open web-cam",command=lambda:os.system('python fire_cam.py')).place(x=260,y=330)

window.mainloop()