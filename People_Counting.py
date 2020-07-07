from Centroid_Tracking_Algo import CentroidTracker
from Trackable_Object import TrackableObject
import imutils
import numpy as np
import cv2
import argparse
import dlib

# Construct the arguments & parse the argument
ap=argparse.ArgumentParser()
ap.add_argument("--prototxt",required=True,help="path to caffe deploy prototxt file")
ap.add_argument("--model", required=True,help="path to caffe pre-trained model")
ap.add_argument("--input_video",required=True,help="path to input video")
ap.add_argument("--skip_frames",type=int,default=30,help="skip frame between detections")
args=vars(ap.parse_args(["--prototxt","MobileNetSSD_deploy.prototxt",
                         "--model","MobileNetSSD_deploy.caffemodel",
                         "--input_video","2.mp4"]))

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load_our pre-trained model
net=cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])

# video initialize
video=cv2.VideoCapture(args["input_video"])

# initialize the frame dimensions
w=None
h=None

# initializer our centroid tracker then initialize list to store each of our dlib correlation tracker
# followed by dictionary to map each unique ID to trackableobject
ct=CentroidTracker(maxDisappeared=40,maxDistance=50)
trackers=[]
trackableObjects= {}

# initialize total no of frame & no of objects move either up or down
totalFrames=0
totalDown=0
totalUp=0

# loop over frame from video capture
while True:
    # grab the frame from video
    ret,frame=video.read()
    if args["input_video"] is not None and frame is None:
        break

    frame=imutils.resize(frame,700)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    if w is None or h is None:
        (h,w)=frame.shape[:2]

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status="Waiting"
    rects=[]

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % args["skip_frames"] ==0:
        # set the status and initialize our new set of object tracker
        status="Detecting"
        trackers=[]

        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob=cv2.dnn.blobFromImage(frame,0.007843,(w,h),127.5)
        net.setInput(blob)
        detections=net.forward()

        # loop over the frame
        for i in np.arange(0, detections.shape[2]):
            # extracted the confidences (i.e. probability) associated with the prediction
            confidence = detections[0,0,i,2]

            # filter out weak predictions
            if confidence > 0.5:
                # extract the index of the class label from the detections list
                idx=int(detections[0,0,i,1])

                # if class label not the person , ignore it
                if CLASSES[idx] !="person":
                    continue

                # compute the bounding box for the object
                box=detections[0,0,i,3:7] *np.array([w,h,w,h])
                (start_x,start_y,end_x,end_y)= box.astype("int")

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker

                tracker=dlib.correlation_tracker()
                rect=dlib.rectangle(int(start_x),int(start_y),int(end_x),int(end_y))
                tracker.start_track(rgb,rect)
                trackers.append(tracker)

    else:
        # we should utilize our object *trackers* rather than
	    # object detectors* to obtain a higher frame processing throughput
        for tracker in trackers:
            # set the status to "tracking" rather then "waiting" or "detecting"
            status="tracking"
            # update the tracker and grab the update positions
            tracker.update(rgb)
            pos=tracker.get_position()

            # unpack the positions object
            start_x=int(pos.left())
            start_y=int(pos.top())
            end_x=int(pos.right())
            end_y=int(pos.bottom())

            # add the bounding box rectangle to the rectangles list
            rects.append((start_x,start_y,end_x,end_y))

    # draw the horizontal line in the centre of the frame once the object cross the line we will update up or down
    cv2.line(frame,(0,h//2,),(w,h//2),(255,0,0),2)

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects=ct.update(rects)

    # loop over the tracked object

    for (objectID,centroid) in objects.items():
        # check to see if track able object exist for a current object ID
        to=trackableObjects.get(objectID,None)
        # if there is no existing track able objects create one
        if to is None:
            to=TrackableObject(objectID,centroid)

        # otherwise, there is a track able object so we can utilize it
        # to determine direction

        else:
            # the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')

            y=[c[1] for c in to.centroids]
            direction=centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # check to see if object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
				# is moving up) AND the centroid is above the center
				# line, count the object
                if direction<0 and centroid[1]<h//2:
                    totalUp +=1
                    to.counted= True

                # if the direction is positive (indicating the object
				# is moving down) AND the centroid is below the
				# center line, count the object

                if direction>0 and centroid[1]>h//2:
                    totalDown +=1
                    to.counted=True

        # store the track able object in dictionary
        trackableObjects[objectID] = to

        # Draw both ID of the object & centroid of the object on output frame
        text="ID {}".format(objectID)
        cv2.putText(frame,text,(centroid[0]-10,centroid[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        cv2.circle(frame,(centroid[0],centroid[1]),4,(0,255,0),-1)

    # the information i will displaying on the frame
    info= [("Up",totalUp),
           ("Down",totalDown),
           ("Status",status)]

    # loop over the info and draw them in frame
    for (i,(k,v)) in enumerate(info):
        text="{} : {}".format(k,v)
        print(text)
        cv2.putText(frame,text,(10,h-((i*20)+10)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    cv2.imshow("frame",frame)
    key=cv2.waitKey(1) & 0xFF
    if key==ord("q"):
        break
    totalFrames +=1

cv2.destroyAllWindows()

























