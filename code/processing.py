import os
import threading

import cv2
import numpy as np
import pyautogui
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage

import project.config

class Thread_out(threading.Thread):
    def __init__(self, img_queue, proc_queue):
        threading.Thread.__init__(self)
        self.status = True
        self.qu = img_queue
        self.qu_img_to_app = proc_queue
        self.EDGE_TYPE = None
        self.cnt = 0

        self.is_haar = False
        self.is_diff = False
        self.trained_file = None
        self.pre_frame = None

        self.is_mean = False

        self.roi_x1 = 0
        self.roi_y1 = 0
        self.roi_x2 = 0
        self.roi_y2 = 0
        self.is_roi_ready = False

        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0

        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0

        self.isStart = False # Alt + tab

        self.roi_frame = None

        self.r_start = 0
        self.r_end = 0
        self.c_start = 0
        self.c_end = 0

    def run(self):
        global Processing_stop
        while self.status:
            if Processing_stop is True:
                project.config.sema2.release()
                project.config.sema0_2.acquire()
            if self.qu.qsize() > 0:
                cnt_edge = 10
                frame = self.qu.get()


                if self.EDGE_TYPE == 'Laplacian':
                    frame = cv2.Laplacian(frame, cv2.CV_8U, ksize=3)
                    # self.updatePlot.emit(frame)
                elif self.EDGE_TYPE == 'Canny':
                    frame = cv2.Canny(frame, 150, 300)
                    cnt_edge = self.sum_edge(frame)

                if self.is_roi_ready is True:
                    pos_ori = (self.roi_x1, self.roi_y1)
                    pos_end = (self.roi_x2, self.roi_y2)
                    cv2.circle(frame, pos_ori, 8, (255, 0, 255), -1)
                    cv2.circle(frame, pos_end, 8, (255, 0, 255), -1)

                    color = (0, 0, 255)
                    frame = cv2.rectangle(frame, pos_ori, pos_end, color, 2)

                if self.is_haar is True:
                    cascade = cv2.CascadeClassifier(self.trained_file) 
                    frame = self.haar(cascade, frame)  

                if self.is_diff is True:
                    frame = self.diff_img(frame) 

                if self.is_mean is True:
                    frame = self.mean_shift(frame)

                if len(frame.shape) < 3:
                    h, w = frame.shape
                    ch = 1
                    img_format = QImage.Format_Grayscale8
                else:
                    h, w, ch = frame.shape
                    img_format = QImage.Format_RGB888
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame = QImage(frame.data, w, h, ch * w, img_format)
                frame = frame.scaled(640, 480, Qt.KeepAspectRatio)

                qu_val = [frame, cnt_edge]

                self.qu_img_to_app.put_nowait(qu_val)

            project.config.sema3.release()

    def mean_shift(self, frame):

        track_window = (self.x, self.y, self.w, self.h)
        roi = frame[self.y:self.y + self.h, self.x:self.x + self.w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) #frame -> roi
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x, y, w, h = track_window
        cx,cy = (2 * x + w) // 2, (2 * y + h) // 2

        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        img2 = cv2.circle(img2,(cx,cy),8,(0,0,255),-1)

        if self.roi_x1 <= cx <= self.roi_x2 and self.roi_y1 <= cy <= self.roi_y2 and not self.isStart:
            self.isStart = True
            print("!!!!!")
            pyautogui.hotkey('command', 'tab')  # Command + Tab function

        return img2

    
    def haar(self, cascade, frame):
        frame = cv2.resize(frame, dsize=None, fx=0.375, fy=0.375)
        # Reading frame in gray scale to process the pattern
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = cascade.detectMultiScale(gray_frame, scaleFactor=1.09,minNeighbors=5, minSize=(5, 5))


        # Drawing green rectangle around the pattern
        for (x, y, w, h) in detections:
            pos_ori = (x, y)
            pos_end = (x + w, y + h)
            color = (0, 255, 0)
            cx = (2 * x + w) // 2
            cy = (2 * y + h) // 2
            cv2.circle(frame,(cx, cy),6,(0,0,255),-1)
            cv2.rectangle(frame, pos_ori, pos_end, color, 2)

            if self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2 and not self.isStart:
                self.isStart = True
                pyautogui.hotkey('command', 'tab')  # Command + Tab 기능

        return frame


    def diff_img(self, frame):
        img = cv2.resize(frame, dsize=None, fx=0.375, fy=0.375)
        current_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

        if self.pre_frame is None:
            self.pre_frame = current_gray.copy()
            return frame
        else:


            diff = cv2.absdiff(self.pre_frame,current_gray)
            _, thresh =  cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
            self.pre_frame = current_gray.copy()


            rsum = np.sum(thresh, axis=0)
            csum = np.sum(thresh, axis=1)

            rsum = rsum >= 1
            csum = csum >= 1

            rpos = np.where(rsum == 1)
            cpos = np.where(csum == 1)

            if len(rpos[0]) == 0 and len(cpos[0]) == 0:
                r_start = self.r_start
                r_end = self.r_end
                c_start = self.c_start
                c_end = self.c_end
            else :
                r_start, r_end = rpos[0][0], rpos[0][-1]
                c_start, c_end= cpos[0][0], cpos[0][-1]
                self.r_start = r_start
                self.r_end = r_end
                self.c_start = c_start
                self.c_end = c_end

            cv2.rectangle(thresh, (r_start, c_start), (r_end, c_end), (124,252,0), 2)
            return thresh


    def set_file(self, fname):
        self.trained_file = os.path.join(cv2.data.haarcascades, fname)  

    def sum_edge(self, frame):
        ratio = 480 / frame.shape[0]
        print(ratio)
        img = cv2.resize(frame, None, fx=ratio, fy=ratio)
        temp = img > 0

        sum_value = temp.sum()
        return sum_value
