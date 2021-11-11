# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 22:23:34 2021

@author: ATHIRA
"""

import streamlit as st
import cv2
import webbrowser
from PIL import Image
import numpy as np 
import streamlit as st 
import argparse
import time
import numpy as np
import cv2
import moviepy.editor as moviepy
import argparse
import time
import numpy as np
import cv2

import utills
import plot

import plot
from PIL import Image
import numpy as np 
import tempfile
confid = 0.5
thresh = 0.5
mouse_pts = []

rad = st.sidebar.selectbox(
    " Platform ",
    ("CrowdAnalysis", "Unattended Baggage Detection","Blind person detection","Track defect")
)    

if rad == "CrowdAnalysis":
    
    st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)

    st.markdown('<p class="big-font">CrowdAnalysis</p>', unsafe_allow_html=True)
    
if rad == "CrowdAnalysis":
    st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)

    st.markdown('<p class="big-font">Browse</p>', unsafe_allow_html=True)    
    
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
    if st.button('output'):
       st.video("crowdd.mp4")
    st.write("OR")      
    
    if st.button('Check with a live video'):
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
            
        # Function to calculate social distancing violations
        def get_mouse_points(event, x, y, flags, param):

            global mouse_pts
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(mouse_pts) < 4:
                    cv2.circle(image, (x, y), 5, (0, 255, 0), 10)
                else:
                    cv2.circle(image, (x, y), 5, (0, 0, 255), 10)

                if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
                    cv2.line(image, (x, y), (mouse_pts[len(
                        mouse_pts)-1][0], mouse_pts[len(mouse_pts)-1][1]), (70, 70, 70), 2)
                    if len(mouse_pts) == 3:
                        cv2.line(image, (x, y),
                                 (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)

                if "mouse_pts" not in globals():
                    mouse_pts = []
                mouse_pts.append((x, y))


        def calculate_social_distancing(vid_path, net, output_dir, output_vid, ln1):

            count = 0
            vs = cv2.VideoCapture(vid_path)

            # Get video height, width and fps
            height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps = int(vs.get(cv2.CAP_PROP_FPS))

            # Set scale for birds eye view
            scale_w, scale_h = utills.get_scale(width, height)

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            # Initialize writer objects
            output_movie = cv2.VideoWriter("crowd.avi", fourcc, fps, (width, height))
            output_movie2 = cv2.VideoWriter("Output2.avi", fourcc, fps, (1920, 1080))
            bird_movie = cv2.VideoWriter("./output_vid/bird_eye_view.avi",
                                         fourcc, fps, (int(width * scale_w), int(height * scale_h)))

            points = []
            global image

            while True:
                # Read frames
                (grabbed, frame) = vs.read()

                if not grabbed:
                    print('here')
                    break

                (H, W) = frame.shape[:2]

                if count == 0:
                    while True:
                        image = frame
                        cv2.imshow("image", image)
                        cv2.waitKey(1)
                        if len(mouse_pts) == 8:
                            cv2.destroyWindow("image")
                            break

                    points = mouse_pts

                src = np.float32(np.array(points[:4]))
                dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
                # Transform perspective using opencv method
                prespective_transform = cv2.getPerspectiveTransform(src, dst)

                # using next 3 points for horizontal and vertical unit length(in this case 6 Feets ~= 180 cm)
                pts = np.float32(np.array([points[4:7]]))
                warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]

                # Calculate distance scale using marked points by user
                distance_w = np.sqrt(
                    (warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
                distance_h = np.sqrt(
                    (warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
                pnts = np.array(points[:4], np.int32)
                cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)

                # Using YOLO v3 model using dnn method
                blob = cv2.dnn.blobFromImage(
                    frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                start = time.time()
                layerOutputs = net.forward(ln1)
                end = time.time()
                boxes = []
                confidences = []
                classIDs = []

                for output in layerOutputs:
                    for detection in output:
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]
                        # detecting humans in frame
                        if classID == 0:

                            if confidence > confid:
                                # Finding bounding boxes dimensions
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")

                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)
                # Applying Non Maximum Suppression to remove multiple bounding boxes around same object
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
                font = cv2.FONT_HERSHEY_PLAIN
                boxes1 = []
                for i in range(len(boxes)):
                    if i in idxs:
                        boxes1.append(boxes[i])
                        x, y, w, h = boxes[i]

                if len(boxes1) == 0:
                    count = count + 1
                    continue

                # Get transformed points using perspective transform
                person_points = utills.get_transformed_points(
                    boxes1, prespective_transform)

                # Get distances between the points
                distances_mat, bxs_mat = utills.get_distances(
                    boxes1, person_points, distance_w, distance_h)

                # Get the risk counts
                risk_count = utills.get_count(distances_mat)

                frame1 = np.copy(frame)

                bird_image = plot.bird_eye_view(
                    frame, distances_mat, person_points, scale_w, scale_h, risk_count)
                img = plot.social_distancing_view(frame1, bxs_mat, boxes1, risk_count)
                if count != 0:

                    bird_movie.write(bird_image)

                    cv2.imshow('Social Distancing Detect', img)
                    output_movie.write(img)
                    output_movie2.write(img)
                    cv2.imwrite(output_dir+"frame%d.jpg" % count, img)
                    cv2.imwrite(output_dir+"bird_eye_view/frame%d.jpg" %
                                count, bird_image)

                count = count + 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                output_movie.write(img)

            vs.release()
            cv2.destroyAllWindows()


        if __name__ == "__main__":

            video_path = 'live.avi'
            model_path = './models/'

            output_dir = './output/'
            output_vid = './output_vid/'

            # load Yolov3 weights

            weightsPath = model_path + "yolov3.weights"
            configPath = model_path + "yolov3.cfg"

            # Initializing yolov3 weights
            net_yl = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
            ln = net_yl.getLayerNames()
            ln1 = [ln[i - 1] for i in net_yl.getUnconnectedOutLayers()]

            cv2.namedWindow("image")
            cv2.setMouseCallback("image", get_mouse_points)
            np.random.seed(42)

            # Start the detection
            calculate_social_distancing(
                video_path, net_yl, output_dir, output_vid, ln1)
            clip = moviepy.VideoFileClip("crowd.avi")
            clip.write_videofile("crowd.mp4")

            st.video("crowd.mp4")
if rad == "Unattended Baggage Detection":
     st.markdown("""
 <style>
 .big-font {
     font-size:30px !important;
 }
 </style>
 """, unsafe_allow_html=True)

     st.markdown('<p class="big-font">Unattended Baggage Detection</p>', unsafe_allow_html=True)
     
 
     
     
     uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
     if st.button('output'):
       st.video("myvide.mp4")
   
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
 
if rad == "Track defect":
  st.title("Track Defect Detection")
# Function to Read and Manupilate Images
  def load_image(img):
       im = Image.open(img)
       image = np.array(im)
       return image

# Uploading the File to the Page
  uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])

# Checking the Format of the page
  if uploadFile is not None:
    # Perform your Manupilations (In my Case applying Filters)
    img = load_image(uploadFile)
    
    image=st.image(img)
    st.write("Image Uploaded Successfully")
    cv2.imwrite('image.png',img)
    st.write("defective")
    cv2.imwrite('image.png',img)
    st.image('image.png',img)
  else:
    st.write("Make sure you image is in JPG/PNG Format.")  
      