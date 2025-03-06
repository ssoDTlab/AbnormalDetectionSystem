import sys
import threading
import time

import cv2

import project.config

class Thread_in(threading.Thread):
    def __init__(self, img_queue):
        threading.Thread.__init__(self)
        self.status = True
        self.capture = None
        self.qu = img_queue
        self.vid = None
        self.frame = None
    def run(self):
        self.capture = cv2.VideoCapture(self.vid)

        if not self.capture.isOpened():
            print("Camera open failed")

        prevTime = 0
        fps = 20
        while self.status:
            global Processing_stop
            if Processing_stop is True:
                project.config.sema1.release()
                project.config.sema0_1.acquire()

            curTime = time.time()  # 현재 시간
            sec = curTime - prevTime
            if sec > 1 / fps:
                prevTime = curTime
                project.config.sema3.acquire()
                ret, frame = self.capture.read()
                self.frame = frame
                if not ret:
                    continue
                self.qu.put(frame)
        sys.exit(-1)