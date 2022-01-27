import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import cv2
import threading
from PyQt5 import QtGui
from tensorflow.keras.models import load_model
import numpy as np
import time
####################################################################모델코드############################################################################
model=load_model('NSWD18-0.2569.hdf5')
column=['Open Eyes','Closed Eyes']
 
####################################################################모델코드############################################################################
form_class = uic.loadUiType("NSWD_GUI.ui")[0]

class NSWD(QMainWindow,form_class):
   
   def __init__(self):
       super().__init__()
       self.setupUi(self)
     
 
   running = False
   def run(self):
       global running
       global roi_scaled
    
       cap = cv2.VideoCapture(0)
 
       self.trans_imgsize=(400,400,3)
       face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
       while running:
           ret, frame = cap.read()
           
           frame=cv2.resize(frame,(400,400))
       
           gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
           frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           faces=face_cascade.detectMultiScale(gray,1.3,5) # scalefactor: and minNeighbors :중복되는 얼굴을 찾을 확률을 줄여준다
           for (x,y,w,h) in faces:
               cv2.rectangle(frame,(x-10,y-50),(x+w+10,y+h+50),(0,0,0),7)
               
               roi_color=frame[x-30:x+w+30,y+15:y+h-15]
               ###########################################################
               ################이 처리를 안해주면 네모가 화면 밖으로 나가면 프로그램이 뻑난다
               if x-50<0 or x+w+50>400 or y-50<0 or y+h+50>400:
                   self.stop()                
               else:
                   
                   roi_scaled=cv2.resize(roi_color,(100,100))
                   
                   roi_scaled_4predict=np.expand_dims(roi_scaled,axis=0) ######나눠주지 않으면 파이큐티를 위한 차원과 모델예측을 위한 차원이 맞지 않게된다. 
                   #print(roi_scaled_4predict.shape)
                   
                   #print(column[model.predict(roi_scaled_4predict).argmax()])

                   print(model.predict(roi_scaled_4predict))
                   #if model.predict(roi_scaled_4predict)[0]>0.2:
                   #    print('Eye Opened')
                   #else:
                   #    print('Eye Closed')
                   #time.sleep(0.2) 
                   fImg = QtGui.QImage(roi_scaled.data, 100,100,300, QtGui.QImage.Format_RGB888)
                   fpixmap = QtGui.QPixmap.fromImage(fImg) ## f img의 경우 else 문 안에 안넣으면 카메라 각도가 잘 안맞춰진상태에서 카메라를 켰을 때 else문을 안거치고 밑으로 빠지므로 not defined 에러가 뜬다.
                   self.facelabel.setPixmap(fpixmap)
         
         
         
           qImg = QtGui.QImage(frame.data, self.trans_imgsize[0], self.trans_imgsize[1], self.trans_imgsize[0]*self.trans_imgsize[2], QtGui.QImage.Format_RGB888)
           
           pixmap = QtGui.QPixmap.fromImage(qImg)
           #fImg = QtGui.QImage(roi_scaled.data, 100,100,300, QtGui.QImage.Format_RGB888)
           #fpixmap = QtGui.QPixmap.fromImage(fImg)
           
         

           self.camlabel.setPixmap(pixmap)
           
     
       cap.release()
 
   
    
       
   def stop(self):
       global running
       running = False
 
   def start(self):
       global running
       running = True
       th = threading.Thread(target=self.run)
       th.start()
   
   
 
   def statuschange(self):
       self.statuslabel.setText('warning')
 
if __name__=="__main__":
   app=QApplication(sys.argv)
   object1=NSWD()
   object1.setWindowTitle('NSWD')
   object1.show()
   sys.exit(app.exec_())
 
 
