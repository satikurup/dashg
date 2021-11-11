# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 13:02:13 2021

@author: ATHIRA
"""

import math
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from centroidtracker import *
import moviepy.editor as moviepy
rad = st.sidebar.selectbox(
    " Platform ",
    ("CrowdAnalysis", "Unattended Baggage Detection","Blind person detection")
)    
ra = st.sidebar.selectbox(
    " Track ",
    ("Track defect")
)    

if rad == "CrowdAnalysis":
    st.title("CrowdAnalysis")
    col1,col2=st.columns([1,1])
    
    with col1:
     
      if st.button('Demo'):
        
       cap=cv2.VideoCapture("myvide.mp4")
       st.video("myvide.mp4")
       cap.release()
    with col2:
     if st.button('live'):
        def save_webcam(outPath,fps,mirror=False):
            # Capturing video from webcam:
            cap = cv2.VideoCapture(0)

            currentFrame = 0

            # Get current width of frame
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            # Get current height of frame
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(outPath, fourcc, fps, (int(width), int(height)))

            while (cap.isOpened()):

                # Capture frame-by-frame
                ret, frame = cap.read()

                if ret == True:
                    if mirror == True:
                        # Mirror the output video frame
                        frame = cv2.flip(frame, 1)
                    # Saves for video
                    out.write(frame)

                    # Display the resulting frame
                    cv2.imshow('frame', frame)
                else:
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):  # if 'q' is pressed then quit
                    break

                # To stop duplicate images
                currentFrame += 1

            # When everything done, release the capture
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        def main():
            save_webcam('live.avi', 30.0,mirror=True)
            
        if __name__ == "__main__":
            main()
            
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg") 

        ## Save all the class names in a list  (80 CLASSES)
        with open("coco.names", "r") as f:     
            classes = [word.strip() for word in f.readlines()] 
            
        ## Get layer names of the network 
        layer_names = net.getLayerNames() 

        ## Determine the output layer names from the YOLO model  
        # (net.getUnconnectedOutLayers() gives the index position of the layers)
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()] 

        print("YOLOv3 LOADED SUCCESSFULLY")

        filename = "live.avi"                         # filename
        cap = cv2.VideoCapture(filename)              # loading video

        # We get the resolution of our video (width and height) and we convert from float to integer
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        # We create VideoWriter object and define the codec. The output is stored in 'output.avi' file.
        out_video = cv2.VideoWriter("o.avi",                                # output name
                                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),  # 4-byte code used to specify the video codec
                                                                                 # (we pass MJPG)
                                    10,                                          # number of frames per second (fps) 
                                    (frame_width, frame_height)                  # frame size
                                    )
        
        # set font and color of text and bounding boxes
        font = cv2.FONT_HERSHEY_PLAIN     # font
        color_g = (0, 255, 0)             # green color
        color_r = (0, 0, 255)             # red color
        color_y = (0,255,255)
        color_w = (255,255,255)
        color_0 = (0,165,255)
        color_b = (255,0,0)
        color_grey = (192,192,192)

        ct = CentroidTracker(maxDisappeared=10)

        def calculateCentroid(xmin,ymin,xmax,ymax):

            xmid = ((xmax+xmin)/2)
            ymid = ((ymax+ymin)/2)
            centroid = (xmid,ymid)

            return xmid,ymid,centroid

        def get_distance(x1,x2,y1,y2):

            distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            
            return distance

        while cap.isOpened():     # while the capture is correctly initialized...

            # We process the video frame-by-frame
            
            ret, img = cap.read()           # we read each frame (img) from the video
                                            # we also retrieve ret, which is a boolean value. 
                                            # ret is True if the frame is read correctly
            
            if ret == True:    # if the frame is read correctly, go on...
                
            
                ## EXTRACT REGION OF INTEREST(ROI)
                roi = img[0:, 0:]           # consider only a slice in pixels of the entire frame 

                height, width, _ = roi.shape    # retrieve height and width from the region of interest
                                                # (we need height and width to build bounding boxes later)

                ## IMAGE PREPROCESSING
                # The cv2.dnn.blobFromImage function returns a blob which is our input image after
                # scaling by a scale factor, and channel swapping.
                # The input image that we need to pass to the Yolo algorithm must be 416x416
                blob = cv2.dnn.blobFromImage(roi, 1/255.0, (416,416), (0,0,0), swapRB=True, crop= False)

                ######################
                ## OBJECT DETECTION ##
                ######################
                
                net.setInput(blob)                      # set blob as input to the network
                outs = net.forward(output_layers)       # runs a forward pass to compute the network output  
               
                person_c=[]
                bag_c=[]

                p_boxes = []
                p_confidences = []
                p_class_ids = []

                for out in outs:            # for each output...
                    for detection in out:               # for each detection...
                        scores = detection[5:]             # array with 80 scores (1 score for each class)
                        class_id = np.argmax(scores)       # take the id of the maximum score
                        confidence = scores[class_id]      # confidence of the class with the maximum score

                        if confidence > 0.5 and class_id == 0:    # if the confidence of the detected object is above the threshold
                                                # we start to create the bounding box...
                            # Object detected
                            center_x = int(detection[0] * width)             # x of the center point
                            center_y = int(detection[1] * height)            # y of the center point
                            w = int(detection[2] * width)                    # width of the detected object
                            h = int(detection[3] * height)                   # height of the detected object

                            # Rectangle coordinates
                            x = int(center_x - w / 2)                         # x of the top left point
                            y = int(center_y - h / 2)                         # y of the top left point

                            p_boxes.append([x, y, w, h])
                            p_confidences.append(float(confidence))
                            p_class_ids.append(class_id)


                ## NMS - NON-MAXIMUM SUPPRESSION
                # We use NMS function in opencv to perform Non-maximum Suppression  
                # The function performs non maximum suppression, given boxes and corresponding confidence scores
                # We give it score threshold and nms threshold as arguments:
                # score_threshold: keep only boxes with a confidence score higher than the threshold
                # nms threshold: threshold used in non maximum suppression (IoU)
                # The function returns indices of bounding boxes survived after NMS.
                p_indexes = cv2.dnn.NMSBoxes(p_boxes, p_confidences, score_threshold = 0.5, nms_threshold = 0.3)
                
                
                ## Draw bounding boxes of the final detected objects 
                p_final_boxes = []
                for i in range(len(p_boxes)):              # for each box...
                    if i in p_indexes:                     # if the bounding box has survived after NMS...  
                
                        x, y, w, h = p_boxes[i] 
                        xmin = x
                        ymin = y
                        xmax = (x + w)
                        ymax = (y + h)  
                        xmid, ymid, p_centroid = calculateCentroid(xmin,ymin,xmax,ymax)
                        #detectedBox.append([xmin,ymin,xmax,ymax,centroid])
                                           # bounding box coordinates
                        person_c.append(p_centroid)
                        label = str(classes[p_class_ids[i]])                         # class label
                        cv2.rectangle(roi, (x, y), (x + w, y + h), color_g, 2)     # drawing rectangular bounding box
                                                                        # (x,y) is the top left corner of the box
                                                                        # (x + w, y + h) is the bottom right corner of the box
                        cv2.putText(roi, label, (x, y - 5), font, 1, color_g, 2)   # class of the detected object 
                        p_final_boxes.append([x, y, x+w, y+h])            # append bounding box to the final boxes list

                b_boxes = []
                b_confidences = []
                b_class_ids = []

                for out in outs:            # for each output...
                    for detection in out:               # for each detection...
                        scores = detection[5:]             # array with 80 scores (1 score for each class)
                        class_id = np.argmax(scores)       # take the id of the maximum score
                        confidence = scores[class_id]      # confidence of the class with the maximum score

                        if confidence > 0.5 and class_id in [26,28]:   # if the confidence of the detected object is above the threshold
                                                # we start to create the bounding box...
                            # Object detected
                            center_x = int(detection[0] * width)             # x of the center point
                            center_y = int(detection[1] * height)            # y of the center point
                            w = int(detection[2] * width)                    # width of the detected object
                            h = int(detection[3] * height)                   # height of the detected object

                            # Rectangle coordinates
                            x = int(center_x - w / 2)                         # x of the top left point
                            y = int(center_y - h / 2)                         # y of the top left point

                            b_boxes.append([x, y, w, h])
                            b_confidences.append(float(confidence))
                            b_class_ids.append(class_id)


                ## NMS - NON-MAXIMUM SUPPRESSION
                # We use NMS function in opencv to perform Non-maximum Suppression  
                # The function performs non maximum suppression, given boxes and corresponding confidence scores
                # We give it score threshold and nms threshold as arguments:
                # score_threshold: keep only boxes with a confidence score higher than the threshold
                # nms threshold: threshold used in non maximum suppression (IoU)
                # The function returns indices of bounding boxes survived after NMS.
                b_indexes = cv2.dnn.NMSBoxes(b_boxes, b_confidences, score_threshold = 0.5, nms_threshold = 0.3)
                
                
                ## Draw bounding boxes of the final detected objects 
                b_final_boxes = []
                for i in range(len(b_boxes)):              # for each box...
                    if i in b_indexes:                     # if the bounding box has survived after NMS...  
                
                        x, y, w, h = b_boxes[i]
                        #x, y, w, h = p_boxes[i] 
                        xmin = x
                        ymin = y
                        xmax = (x + w)
                        ymax = (y + h)  
                        xmid, ymid, centroid = calculateCentroid(xmin,ymin,xmax,ymax)
                                               # bounding box coordinates
                        bag_c.append(centroid)
                        label = str(classes[b_class_ids[i]])                         # class label
                        cv2.rectangle(roi, (x, y), (x + w, y + h), color_b, 2)     # drawing rectangular bounding box
                                                                        # (x,y) is the top left corner of the box
                                                                        # (x + w, y + h) is the bottom right corner of the box
                        cv2.putText(roi, label, (x, y - 5), font, 1, color_b, 2)   # class of the detected object 
                        b_final_boxes.append([x, y, x+w, y+h])            # append bounding box to the final boxes list

                
                #####################
                ## OBJECT TRACKING ##
                #####################
                p_c=[]
                b_c=[]
                p_objects = ct.update(p_final_boxes)   # we pass the list with final boxes to the tracker in order to have 
                                                   # a unique id for each detection and its specific centroid coordinates

                ## Draw the ID of the detected and tracked object
                for (p_objectID, p_centroid) in p_objects.items():    # for each tracked object...
                    text_p= str(p_objectID +1)                                # unique id of the tracked object
                    cv2.putText(roi, text_p, (p_centroid[0], p_centroid[1]), font, 2, color_y, 2)   # text
                    p_c.append(p_centroid)
                    #cv2.circle(roi, (p_centroid[0], p_centroid[1]), 4, (0, 255, 0), -1) # draw the centroid of tracked object

                b_objects = ct.update(b_final_boxes)   # we pass the list with final boxes to the tracker in order to have 
                                                   # a unique id for each detection and its specific centroid coordinates

                ## Draw the ID of the detected and tracked object
                for (b_objectID, b_centroid) in b_objects.items():    # for each tracked object...
                    text_b = str(b_objectID +1)                                # unique id of the tracked object
                    cv2.putText(roi, text_b, (b_centroid[0], b_centroid[1]), font, 2, color_r, 2) 
                    b_c.append(b_centroid)  # text
                    #cv2.circle(roi, (centroid[0], centroid[1]), 4, (0, 255, 0), -1) # draw the centroid of tracked object            
                #output = [[(a1, a2), (b1, b2)] for (a1, a2) in person_c
                            #for (b1, b2) in bag_c if get_distance(a1,b1,a2,b2) <= 50] 
                #print(output)  
                #if len(output) > 0:
                 #   print("Baggage_Owner pair", output)
                
                #test_keys = ["Rash", "Kil", "Varsha"]
                #test_values = [1, 4, 5]
          
        # Printing original keys-value lists
                #print ("Original key list is : " + str(test_keys))
                #print ("Original value list is : " + str(test_values))
          
        # using naive method
        # to convert lists to dictionary                            
                cv2.imshow("out", roi)
                out_video.write(img)        # the frame is saved for the final video
                
                if cv2.waitKey(1) & 0xFF == ord('q'):              # if exit button, break and close
                    break
            
            
            else:   # if the frame is not read correctly, break...
                break
              
        # Release everything when job is finished
        cap.release()
        #clip = moviepy.VideoFileClip("o.avi")
        #clip.write_videofile("o.mp4")
        st.video("o.avi") 
        cv2.destroyAllWindows()
  