from threading import Thread
from flask import Flask, jsonify, request, Response
from flask_restful import Resource, Api
from Centroid_Tracking_Algo import CentroidTracker
from Trackable_Object import TrackableObject
import imutils
import numpy as np
import cv2
import dlib
import requests

"""app = Flask(__name__)
api = Api(app)"""


class VideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        (self._grabbed, self._frame) = self.stream.read()
        self._stopped = False

    def start(self):
        Thread(target=self.update).start()
        return self

    def update(self):
        while self.stream.isOpened():
            if self._stopped:
                return
            self._grabbed, self._frame = self.stream.read()

    def read(self):
        return self._frame

    def stop(self):
        self._stopped = True
        self.stream.release()

    @property
    def stopped(self):
        return self._stopped

    @property
    def grabbed(self):
        return self._grabbed


class PersonCount:
    def __init__(self):
        self.bus_cam_id = None
        self.cam_stream = None
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
        self.w = None
        self.h = None
        self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        self.trackers = []
        self.trackableObjects = {}
        self.totalFrames = 0
        self.totalDown = 0
        self.totalUp = 0
        self.status = None

    def post(self):

        # postedData = request.get_json()
        postedData = {
            "bus_cam_id": 123,
            "cam_stream": "1.mp4"
        }

        try:
            self.bus_cam_id = postedData["bus_cam_id"]
            self.cam_stream = postedData["cam_stream"]
        except:
            pass

        if self.bus_cam_id is None:
            retJson = {
                "status": 404,
                "message": "Bus Camera ID Not Found"
            }
            return retJson

        if self.cam_stream is None:
            retJson = {
                "status": 404,
                "message": "Stream Not Availiable"
            }

            return retJson

        self.calculate_person()

        """retJson = {
            "congestion_level": num_of_people,
            "status": 200,
            "message": "Api runs successfully"
        }"""

        return self.totalUp

    def calculate_person(self):
        vs = VideoStream(self.cam_stream).start()
        while not vs.stopped:
            frame = vs.read()
            if not vs.grabbed:
                break
            frame = imutils.resize(frame, 700)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.w is None or self.h is None:
                (self.h, self.w) = frame.shape[:2]
                print(self.h, self.w)

            self.status = "waiting"
            rects = []

            if self.totalFrames % 10:
                print("Hi")
                status = "Detecting"
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.w, self.h), 127.5)
                self.net.setInput(blob)
                detections = self.net.forward()

                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        idx = int(detections[0, 0, i, 1])
                        if self.CLASSES[idx] != "person":
                            continue

                        box = detections[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])
                        (start_x, start_y, end_x, end_y) = box.astype("int")

                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(int(start_x), int(start_y), int(end_x), int(end_y))
                        tracker.start_track(rgb, rect)
                        self.trackers.append(tracker)

                else:
                    for tracker in self.trackers:
                        self.status = "tracking"
                        tracker.update(rgb)
                        pos = tracker.get_position()

                        start_x = int(pos.left())
                        start_y = int(pos.top())
                        end_x = int(pos.right())
                        end_y = int(pos.bottom())

                        rects.append((start_x, start_y, end_x, end_y))

            cv2.line(frame, (0, self.h // 2,), (self.w, self.h // 2), (255, 0, 0), 2)

            objects = self.ct.update(rects)

            for (objectID, centroid) in objects.items():
                to = self.trackableObjects.get(objectID, None)
                if to is None:
                    to = TrackableObject(objectID, centroid)
                else:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)

                    if not to.counted:
                        if direction < 0 and centroid[1] < self.h // 2:
                            self.totalUp += 1
                            to.counted = True

                        if direction > 0 and centroid[1] > self.h // 2:
                            self.totalDown += 1
                            to.counted = True

                self.trackableObjects[objectID] = to
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            info = [("Up", self.totalUp),
                    ("Down", self.totalDown),
                    ("Status", self.status)]

            for (i, (k, v)) in enumerate(info):
                text = "{} : {}".format(k, v)
                print(text)
                cv2.putText(frame, text, (10, self.h - ((i * 20) + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            self.totalFrames += 5
        vs.stop()


# api.add_resource(PersonCount, '/person-count')

if __name__ == "__main__":
    pc = PersonCount()
    pc.post()