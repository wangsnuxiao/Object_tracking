import sys
import torch
import numpy as np
import cv2
from PIL import Image
import Kalman_Filter as KF 



def ballbox(img,model):
    results = model(img)
    arr=results.xyxy[0].cpu().detach().numpy()
    balls_index=np.where(arr[:,-1]==49)

    boxes=arr[balls_index,0:-2].astype(np.int16)
    return boxes[0]


input = sys.argv[1]
output = sys.argv[2]
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')# or yolov5m, yolov5l, yolov5x, custom




# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2




pathvid= input # video path
cap = cv2.VideoCapture(pathvid)
ret, frame = cap.read()
(w,h,_)=frame.shape
cap = cv2.VideoCapture(pathvid)
result = cv2.VideoWriter( output, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, (h,w))


#create kalamn filer objects :
#########################
kf1=KF.K_Filter([0,0],[0,0])
kf2=KF.K_Filter([0,0],[0,0]) 
kfarr=[kf1,kf2]


#########################


centers=np.zeros((2,2))


# Check if video opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):




  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # Display the resulting frame
    boxes=ballbox(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB ),model)
    
   

    #prediction 
    ##############
    for  j,i in enumerate(kfarr):
        pred=i.predictoutput(.1)
        centers[j]=pred
    ################



    # match the trackers  with the detected objects 
    ##########################
    kfpredcop=centers.copy()
    
    for j,i in enumerate(boxes):
        start_point=i[[0,1]]
        end_point=i[[2,3]]
        c=(start_point+end_point)/2
        
       

            
        kfarr[j].set_state(c,[0,0])
        

        diff=c-kfpredcop
        dis=np.sum((diff*diff),axis=1)
        indx=np.argmin(dis)
        kfpredcop[indx]=[9999999,9999999]

        centers[indx]=c

        
        #print(start_point,end_point)
        frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
    ##########################

    # update the kalman filter using the measurments 
    #########################
    for  j,i in enumerate(kfarr):
        i.predict(.1)
        i.update(np.array(centers[j]),.1*np.eye(2))
        
        frame = cv2.circle(frame, i.X[[0,1]].astype(np.int16), 3, (0,255//(j+1),0), thickness)

    #########################
    result.write(frame) 

    

    cv2.imshow('Frame',frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()




