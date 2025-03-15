import os
import sys
import time
import cv2
import numpy as np
import pyautogui
import queue
import tensorflow as tf
import project.config
import project.capture
import project.processing
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QMainWindow, QLabel, QScrollBar, QFrame, QLineEdit, QSizePolicy, QPushButton, QHBoxLayout, \
    QComboBox, QVBoxLayout, QGridLayout, QRadioButton, QWidget
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # Title and dimensions
        self.setWindowTitle("Abnormal detection")
        self.setGeometry(0, 0, 1200, 500)


        self.cnt_edges = []
        self.flag = False
        self.isPlay = False
        self.y_max = 0

        # Draw a real-time graph of the video where Edge is detected
        self.canvas = FigureCanvas(Figure(figsize=(0, 0.5)))
        self.axes = self.canvas.figure.subplots()
        n_data = 50
        self.xdata = list(range(n_data))
        self.axes.set_xticks(self.xdata, [])
        self.ydata = [0 for i in range(n_data)]

        # self.m_main_img = None
        self.m_proc_img = None
        self.MODE_VIDEO = False
        self.th_in = None
        self.th_out = None
        self.EDGE_TYPE = None
        self.previous_plot = None
        self.labeling_capture = None
        self.labeling_Ground_truth = []
        self.cnt = 0
        self.mouse_cnt = 0  

        # roi
        self.roi_x1 = 0
        self.roi_y1 = 0
        self.roi_x2 = 0
        self.roi_y2 = 0


        self.label_image = QLabel(self)
        self.img_size = QSize(640, 480)
        self.label_image.setFixedSize(self.img_size)

        self.scroll_bar = QScrollBar(Qt.Horizontal) 
        self.scroll_bar.setMinimum(0)  
        self.scroll_bar.setMaximum(100)  
        self.scroll_bar.setValue(0)  
        self.scroll_bar.setVisible(False)
        self.scroll_bar.valueChanged.connect(self.change_frame)

        self.label_text_scroll = QLabel("Index Number: ")

        self.label_idx_scroll = QLabel()
        self.label_idx_scroll.setFrameStyle(QFrame.Box | QFrame.Raised)

        self.label_text_labeling = QLabel("Labeling(0 or 1): ")
        self.edit_text_labeling = QLineEdit()  #
        self.edit_text_labeling.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.edit_text_labeling.setFixedWidth(20)
        self.edit_text_labeling.setMaxLength(1)

        # Section labeling button
        self.button_start_labeling = QPushButton("section(Start)")
        self.button_end_labeling = QPushButton("section(End)")
        self.button_end_labeling.setEnabled(False)

        self.edit_section_label = QLineEdit()
        self.edit_section_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.edit_section_label.setFixedWidth(20)
        self.edit_section_label.setMaxLength(1)

        self.button_save_label = QPushButton("Save Label")
        self.button_training = QPushButton("Training")
        self.button_REC = QPushButton("REC")
        self.button_REC.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        layout_save_training = QHBoxLayout()
        layout_save_training.addWidget(self.button_REC)
        layout_save_training.addWidget(self.button_save_label)
        layout_save_training.addWidget(self.button_training)

        # mean shift 
        self.button_mean_shift = QPushButton("Mean Shift")
        self.button_mean_shift.setEnabled(False)
        layout_save_training.addWidget(self.button_mean_shift
                                       )
        # (alt + tab) checking
        self.button_restart = QPushButton("Re+ALT+TAB")  # mission
        self.button_restart.setEnabled(False)  # mission
        layout_save_training.addWidget(self.button_restart) # mission

        #ROI
        self.button_roi = QPushButton("ROI")
        self.button_roi.setEnabled(False)
        layout_save_training.addWidget(self.button_roi)

        layout_scroll_idx = QHBoxLayout()
        layout_scroll_idx.addWidget(self.label_text_scroll, alignment=Qt.AlignmentFlag.AlignRight)
        layout_scroll_idx.addWidget(self.label_idx_scroll)
        layout_scroll_idx.addWidget(self.label_text_labeling, alignment=Qt.AlignmentFlag.AlignRight)
        layout_scroll_idx.addWidget(self.edit_text_labeling)
        layout_scroll_idx.addWidget(self.button_start_labeling)
        layout_scroll_idx.addWidget(self.edit_section_label)
        layout_scroll_idx.addWidget(self.button_end_labeling)

        self.label_clip_image = QLabel(self)

        self.combobox_haar = QComboBox()  
        for xml_file in os.listdir(cv2.data.haarcascades):  
            if xml_file.endswith(".xml"):  
                self.combobox_haar.addItem(xml_file)  
        self.button_haar_start = QPushButton("Start(Face_Det)")  
        self.button_haar_stop = QPushButton("Stop/Close")  

        self.button_haar_start.setEnabled(False)  
        self.button_haar_stop.setEnabled(False)  

        self.button_diff_img = QPushButton("Diff Image")  
        self.button_diff_img.setEnabled(False)  



        layout_haar = QHBoxLayout() 
        layout_haar.addWidget(self.combobox_haar)  
        layout_haar.addWidget(self.button_haar_start)  
        layout_haar.addWidget(self.button_haar_stop)  

        layout_clip_haar = QVBoxLayout() 
        layout_clip_haar.addWidget(self.canvas)
        layout_clip_haar.addWidget(self.label_clip_image)  
        layout_clip_haar.addLayout(layout_haar) 
        layout_clip_haar.addWidget(self.button_diff_img)  
        
        # init widgets for perspective image
        self.m_pos_cnt = 0

        # init widgets for edge detection
        self.label_filter = QLabel("Filter type")
        self.button_edge_detection = QPushButton("Edge Detection")
        self._edgeType_combo_box = QComboBox()
        self._edgeType_combo_box.addItem("None")
        self._edgeType_combo_box.addItem("Sobel_XY")
        self._edgeType_combo_box.addItem("Scharr_X")
        self._edgeType_combo_box.addItem("Scharr_Y")
        self._edgeType_combo_box.addItem("Laplacian")
        self._edgeType_combo_box.addItem("Canny")

        # layout for edge detection
        edge_layout = QHBoxLayout()
        edge_layout.addWidget(self.label_filter, alignment=Qt.AlignmentFlag.AlignRight)
        edge_layout.addWidget(self._edgeType_combo_box)
        edge_layout.addWidget(self.button_edge_detection)

        # Perspective Image
        self.button_init_pos = QPushButton("Initialize Pos")
        self.button_perspective_img = QPushButton("Perspective Image")

        # pos1~pos4 label
        self.label_pos1 = QLabel("Pos1")
        self.label_pos2 = QLabel("Pos2")
        self.label_pos3 = QLabel("Pos3")
        self.label_pos4 = QLabel("Pos4")

        # pos1~pos4 each (x,y)
        self.Ledit_x1 = QLineEdit()
        self.Ledit_y1 = QLineEdit()
        self.Ledit_x2 = QLineEdit()
        self.Ledit_y2 = QLineEdit()
        self.Ledit_x3 = QLineEdit()
        self.Ledit_y3 = QLineEdit()
        self.Ledit_x4 = QLineEdit()
        self.Ledit_y4 = QLineEdit()

        perspective_grid = QGridLayout()
        perspective_grid.addWidget(self.label_pos1, 0, 0)
        perspective_grid.addWidget(self.Ledit_x1, 0, 1)
        perspective_grid.addWidget(self.Ledit_y1, 0, 2)

        perspective_grid.addWidget(self.label_pos2, 0, 3)
        perspective_grid.addWidget(self.Ledit_x2, 0, 4)
        perspective_grid.addWidget(self.Ledit_y2, 0, 5)

        perspective_grid.addWidget(self.label_pos3, 1, 0)
        perspective_grid.addWidget(self.Ledit_x3, 1, 1)
        perspective_grid.addWidget(self.Ledit_y3, 1, 2)

        perspective_grid.addWidget(self.label_pos4, 1, 3)
        perspective_grid.addWidget(self.Ledit_x4, 1, 4)
        perspective_grid.addWidget(self.Ledit_y4, 1, 5)

        perspective_layout = QHBoxLayout()
        perspective_layout.addWidget(self.button_init_pos)
        perspective_layout.addWidget(self.button_perspective_img)

        perspective_grid_and_layout = QHBoxLayout()
        perspective_grid_and_layout.addLayout(perspective_grid)
        perspective_grid_and_layout.addLayout(perspective_layout)

        # Geometry
        self.label_geometry = QLabel("Geometry type:")

        self.geometry_combo_box = QComboBox()
        self.geometry_combo_box.addItem("flip")
        self.geometry_combo_box.addItem("translation")
        self.geometry_combo_box.addItem("rotation")

        self.button_geometry_img = QPushButton("Geometry Image")

        # layout for geometry
        geometry_layout = QHBoxLayout()
        geometry_layout.addWidget(self.label_geometry, alignment=Qt.AlignmentFlag.AlignRight)
        geometry_layout.addWidget(self.geometry_combo_box)
        geometry_layout.addWidget(self.button_geometry_img)


        # Load image buttons layout
        self.radiobutton_1 = QRadioButton("Image")
        self.radiobutton_2 = QRadioButton("Video")
        self.radiobutton_3 = QRadioButton("Webcam")
        self.radiobutton_1.setChecked(True)

        layout_loading_type = QHBoxLayout()
        layout_loading_type.addWidget(self.radiobutton_1)
        layout_loading_type.addWidget(self.radiobutton_2)
        layout_loading_type.addWidget(self.radiobutton_3)

        self.button_load_Img = QPushButton("Load Image")
        self.edit = QLineEdit("../videos/moving_dark.mp4")

        self.button_binary_Img = QPushButton("Binary Image")
        self.button_labeling = QPushButton("Labeling")

        self.button_load_Img.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.button_binary_Img.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.edit.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.button_labeling.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.edit)
        bottom_layout.addLayout(layout_loading_type)
        bottom_layout.addWidget(self.button_load_Img)
        bottom_layout.addWidget(self.button_binary_Img)
        bottom_layout.addWidget(self.button_labeling)

        layout_img_scroll = QVBoxLayout()
        layout_img_scroll.addWidget(self.label_image)
        layout_img_scroll.addWidget(self.scroll_bar)
        layout_img_scroll.addLayout(layout_scroll_idx)
        layout_img_scroll.addLayout(layout_save_training)

        # layout for image and graph
        layout_img_canvas = QHBoxLayout()
        layout_img_canvas.addLayout(layout_img_scroll)
        layout_img_canvas.addLayout(layout_clip_haar) 

        # Main layout
        layout = QVBoxLayout()
        layout.addLayout(layout_img_canvas)
        layout.addLayout(bottom_layout)
        layout.addLayout(edge_layout)
        layout.addLayout(perspective_grid_and_layout)
        layout.addLayout(geometry_layout)

        # Central widget
        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Connections
        self.button_load_Img.clicked.connect(self.load_img_func)
        self.button_binary_Img.clicked.connect(self.greetings_binary)
        self.button_REC.clicked.connect(self.recording)
        self.button_labeling.clicked.connect(self.labeling)
        self.button_edge_detection.clicked.connect(self.method_edge_detection)
        self.button_init_pos.clicked.connect(self.method_init_pos)
        self.button_perspective_img.clicked.connect(self.method_perspective_image)
        self.button_geometry_img.clicked.connect(self.method_geometry)

        self.button_restart.clicked.connect(self.restart_alt_tab) 
        self.button_roi.clicked.connect(self.roi) 
        self.button_diff_img.clicked.connect(self.diff_img_start)  
        self.combobox_haar.currentTextChanged.connect(self.set_haar_model)  
        self.button_haar_start.clicked.connect(self.start_haar)  
        self.button_haar_stop.clicked.connect(self.kill_thread)  

        self.button_mean_shift.clicked.connect(self.start_mean_shift) 

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.display_video_stream_v2)

    def method_init_pos(self):
        self.m_pos_cnt = 0

        self.Ledit_x1.setText("")
        self.Ledit_y1.setText("")

        self.Ledit_x2.setText("")
        self.Ledit_y2.setText("")

        self.Ledit_x3.setText("")
        self.Ledit_y3.setText("")

        self.Ledit_x4.setText("")
        self.Ledit_y4.setText("")

        self.load_img_func()

    def method_perspective_image(self):
        rows, cols = self.m_proc_img.shape[:2]  
        x1 = self.Ledit_x1.text()
        y1 = self.Ledit_y1.text()

        x2 = self.Ledit_x2.text()
        y2 = self.Ledit_y2.text()

        x3 = self.Ledit_x3.text()
        y3 = self.Ledit_y3.text()

        x4 = self.Ledit_x4.text()
        y4 = self.Ledit_y4.text()


        pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        pts2 = np.float32([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]])
        Mat1 = cv2.getPerspectiveTransform(pts1, pts2)
        r_image = cv2.warpPerspective(self.m_proc_img, Mat1, (cols, rows))

        self.m_proc_img = r_image
        self.update_image(r_image)

    def method_geometry(self):
        ori_img = self.m_proc_img.copy()
        index = self.geometry_combo_box.currentIndex()
        out_img = None
        if index == 0:  # flip
            f_image_p1 = cv2.flip(ori_img, 1)
            out_img = f_image_p1
        elif index == 1:  # translation
            rows, cols = ori_img.shape[:2]  
            Mat = np.float32([[1, 0, 50],
                              [0, 1, 20]])
            t_image = cv2.warpAffine(ori_img, Mat, (cols, rows),
                                     borderMode=cv2.BORDER_REFLECT)
            out_img = t_image
        elif index == 2:  # rotation
            rows, cols = ori_img.shape[:2] 
            Mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1.0)
            r_image = cv2.warpAffine(ori_img, Mat, (cols, rows),
                                     borderMode=cv2.BORDER_REPLICATE)
            out_img = r_image
        self.update_image(out_img)

    def greetings_binary(self):
        path = self.edit.text()

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, dst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        self.update_image(dst)

    def start_mean_shift(self):
        # cap = cv2.VideoCapture(0)
        # take first frame of the video
        # ret, frame = cap.read()
        # setup initial location of window

        (self.th_out.x, self.th_out.y, self.th_out.w, self.th_out.h) = cv2.selectROI('orange', self.th_in.frame)
        self.th_out.is_mean = True
        cv2.destroyAllWindows()

    def roi(self):
        self.th_out.roi_x1 = int(self.roi_x1 * (720 / 640)) + int(self.roi_x2* (720 / 640))
        self.th_out.roi_y1 = int(self.roi_y1 * (1280 / 480))
        self.th_out.roi_x2 = int(self.roi_x2 * (720 / 640)) + int(self.roi_x2 * (720 / 640))
        self.th_out.roi_y2 = int(self.roi_y2 * (1280 / 480))

        self.th_out.x1 = self.roi_x1
        self.th_out.y1 = self.roi_y1
        self.th_out.x2 = self.roi_x2
        self.th_out.y2 = self.roi_y2

        self.th_out.is_roi_ready = True

    def restart_alt_tab(self):
        self.th_out.isStart = False

    def mousePressEvent(self, event):
        if self.MODE_VIDEO is True:
            label_pos = self.label_image.geometry().getCoords()

            x1, y1, x2, y2 = label_pos
            x = event.position().x() - x1
            y = event.position().y() - y1
            x, y = int(x), int(y)


            if self.mouse_cnt == 0 :
                self.roi_x1 = x
                self.roi_y1 = y
                self.mouse_cnt += 1
            elif self.mouse_cnt == 1 :
                self.roi_x2 = x
                self.roi_y2 = y
                self.mouse_cnt = 0
            return

        x = event.position().x() - self.label_image.x()  
        y = event.position().y() - self.label_image.y()  
        x, y = int(x), int(y)

        if self.m_pos_cnt == 4:
            return
        if self.m_pos_cnt == 0:
            self.Ledit_x1.setText(f"{x}")
            self.Ledit_y1.setText(f"{y}")
            cv2.circle(self.m_proc_img, (x, y), 5, (255, 0, 0), -1)
            self.m_pos_cnt += 1
        elif self.m_pos_cnt == 1:
            self.Ledit_x2.setText(f"{x}")
            self.Ledit_y2.setText(f"{y}")
            cv2.circle(self.m_proc_img, (x, y), 5, (0, 255, 0), -1)
            self.m_pos_cnt += 1
        elif self.m_pos_cnt == 2:
            self.Ledit_x3.setText(f"{x}")
            self.Ledit_y3.setText(f"{y}")
            cv2.circle(self.m_proc_img, (x, y), 5, (0, 0, 255), -1)
            self.m_pos_cnt += 1
        elif self.m_pos_cnt == 3:
            self.Ledit_x4.setText(f"{x}")
            self.Ledit_y4.setText(f"{y}")
            cv2.circle(self.m_proc_img, (x, y), 5, (0, 255, 255), -1)
            self.m_pos_cnt += 1

        self.update_image(self.m_proc_img)

    def start_haar(self):
        self.button_haar_start.setEnabled(False)  
        self.button_haar_stop.setEnabled(True)  
        self.th_out.set_file(self.combobox_haar.currentText())  
        self.th_out.is_haar = True  
        
    def set_haar_model(self, text):
        self.th_out.set_file(text)  


    def diff_img_start(self):
        print("Diff image...")
        self.th_out.is_diff = True  

    def training_perceptron(self):
        print("Training...")
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(3, activation='relu', input_shape=(1,)))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        # model.add(tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(1,)))
        # optimizer = tf.keras.optimizers.SGD(lr=0.00001)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.9, momentum=0.0)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        path_split = self.edit.text().split('/')
        self.file_name = path_split[-1].split('.')
        train_y = np.load(f"{self.file_name[0]}.npy")
        print("train y len: ", len(train_y))
        print("train y size: ", train_y.size)
        train_x = np.array([[]])
        for i in range(len(self.frame_list)):
            frame = cv2.Canny(self.frame_list[i], 150, 300)

            ratio = 480 / frame.shape[0]
            img = cv2.resize(frame, None, fx=ratio, fy=ratio)

            temp = img > 0

            cnt_edge = temp.sum()
            train_x = np.append(train_x, cnt_edge)
        train_x = train_x[:, np.newaxis]
        # train_y = train_y[:,np.newaxis]

        # scaler = MinMaxScaler()
        # scaler.fit(train_x)
        # train_x = scaler.transform(train_x)

        model.fit(train_x, train_y, epochs=2000, batch_size=50, shuffle=True)
        test_loss, test_acc = model.evaluate(train_x, train_y)

        print('테스트 정확도:', test_acc)

        model.save('model_ex6.h5')

        # model = tf.keras.models.load_model('model_ex6.h5')

    def save_label(self):
        print("save label")
        save_Ground_truth = np.array(self.labeling_Ground_truth)
        path_split = self.edit.text().split('/')
        self.file_name = path_split[-1].split('.')
        np.save(f"./{self.file_name[0]}.npy", save_Ground_truth)

    def set_section_start(self):
        print("section start")
        self.start_idx = self.scroll_bar.sliderPosition()
        self.button_end_labeling.setEnabled(True)
        self.button_start_labeling.setEnabled(False)

    def set_section_end(self):
        print("section end")
        idx_i = self.start_idx
        idx_j = self.scroll_bar.sliderPosition()

        if idx_j > idx_i:
            label_value = [int(self.edit_section_label.text()) for i in range(idx_j - idx_i + 1)]
            self.labeling_Ground_truth[idx_i:idx_j + 1] = label_value
        elif idx_j < idx_i:
            label_value = [int(self.edit_section_label.text()) for i in range(idx_i - idx_j + 1)]
            self.labeling_Ground_truth[idx_j:idx_i + 1] = label_value

        self.button_end_labeling.setEnabled(False)
        self.button_start_labeling.setEnabled(True)

    def set_label(self):

        if self.edit_text_labeling.text() == '0' or self.edit_text_labeling.text() == '1':
            self.labeling_Ground_truth[self.scroll_bar.sliderPosition()] = int(self.edit_text_labeling.text())
            print("changed label: ", self.labeling_Ground_truth[self.scroll_bar.sliderPosition()])

        else:
            print("Warning: input only [0 or 1]")

    def change_frame(self):
        cur_idx_of_scroller = self.scroll_bar.sliderPosition()
        # print("changed frame: ", cur_idx_of_scroller)
        self.update_image(self.frame_list[cur_idx_of_scroller])
        self.label_idx_scroll.setText(f"{cur_idx_of_scroller}")
        self.edit_text_labeling.setText(f"{self.labeling_Ground_truth[cur_idx_of_scroller]}")

    def labeling(self):
        print("labeling")
        self.kill_thread()
        self.labeling_capture = cv2.VideoCapture(self.edit.text())

        ret = True
        cnt = 0
        self.frame_list = []
        while ret:
            ret, frame = self.labeling_capture.read()

            if not ret:
                break
            cnt += 1
            self.frame_list.append(frame)
            self.labeling_Ground_truth.append(0)

        print("Loading video is complete")
        print(f"The number of frame: {len(self.frame_list)}  Ground-truth len: {len(self.labeling_Ground_truth)}")
        self.scroll_bar.setMaximum(len(self.frame_list) - 1)  
        self.scroll_bar.setVisible(True)

        # index number QLabel in scroll and QLineEdit in Labeling set initial values
        self.label_idx_scroll.setText("0")
        self.edit_text_labeling.setText(f"{self.labeling_Ground_truth[0]}")

        self.labeling_capture.release()
        self.update_image(self.frame_list[0])

        self.edit_text_labeling.editingFinished.connect(self.set_label)
        self.button_start_labeling.clicked.connect(self.set_section_start)
        self.button_end_labeling.clicked.connect(self.set_section_end)
        self.button_save_label.clicked.connect(self.save_label)
        self.button_training.clicked.connect(self.training_perceptron)

    def recording(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Camera open failed!")
            sys.exit()

        w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # *'DIVX' == 'D', 'I', 'V', 'X'
        delay = round(1000 / fps)

        out = cv2.VideoWriter('output_1.avi', fourcc, fps, (w, h))

        if not out.isOpened():
            print('File open failed!')
            cap.release()
            sys.exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            out.write(frame)
            self.update_image(frame)
            cv2.imshow('frame', frame)

            if cv2.waitKey(delay) == 27:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def closeEvent(self, event):
        self.kill_thread()

    def setup_camera(self, vid):
        self.capture = cv2.VideoCapture(vid)

        if not self.capture.isOpened():
            print("Camera open failed")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size.height())

        self.timer.start(50)

    def normalize_cnt_edges(self,cnt_edge):
        self.cnt_edges.append(cnt_edge)
        if not self.cnt_edges:
            return []

        min_val = min(self.cnt_edges)
        max_val = max(self.cnt_edges)

        if min_val != max_val :
            normalized_value = (cnt_edge - min_val) / (max_val - min_val)
            return normalized_value
        return -1

    def switch_screen(self):
        print("switch")
        pyautogui.hotkey('command', 'tab')

    def update_plot2(self, sum_value):

        self.ydata = self.ydata[1:] + [sum_value]

        if sum_value > self.y_max:
            self.y_max = sum_value
            self.axes.set_ylim([0, self.y_max + 10])
        if self.previous_plot is None:
            self.previous_plot = self.axes.plot(self.xdata, self.ydata, 'r')[0]
        else:
            self.previous_plot.set_ydata(self.ydata)

        global Processing_stop
        Processing_stop = True
        project.config.sema1.acquire()
        project.config.sema2.acquire()

        prevTime = time.time()  
        self.canvas.draw()
        Processing_stop = False
        project.config.sema0_1.release()
        project.config.sema0_2.release()
        curTime = time.time()  
        sec = curTime - prevTime
        #print(sec % 60)


    def display_video_stream_v2(self):
        if self.qu_img_to_app.empty() is False:

            qu_val = self.qu_img_to_app.get_nowait()

            frame = qu_val[0]
            cnt_edge = qu_val[1]

            self.update_image2(frame)

            # print(self.qu_img_to_app.qsize())
            if self.EDGE_TYPE == 'Canny':
                if cnt_edge is not None:
                    normalized_value = self.normalize_cnt_edges(cnt_edge)
                    print(normalized_value)
                    if 0.89 > normalized_value > 0.80: self.flag = True
                    if 1.0 > normalized_value > 0.93 and self.flag and not self.isPlay:
                        self.isPlay = True
                        pyautogui.hotkey('command', 'tab')
                        pyautogui.hotkey('command', 'tab')
                self.update_plot2(cnt_edge)

    def update_image2(self, scaled_img):
        # Creating and scaling QImage
        self.label_image.setFixedSize(scaled_img.width(), scaled_img.height())
        self.label_image.setPixmap(QPixmap.fromImage(scaled_img))

    def method_edge_detection(self):
        if self.MODE_VIDEO is True:
            if self._edgeType_combo_box.currentText() == 'None':
                self.th_out.EDGE_TYPE = None
                return
            elif self._edgeType_combo_box.currentText() == 'Canny':
                self.th_out.EDGE_TYPE = 'Canny'
                self.EDGE_TYPE = 'Canny'
                return
            elif self._edgeType_combo_box.currentText() == 'Laplacian':
                self.th_out.EDGE_TYPE = 'Laplacian'
                self.EDGE_TYPE = 'Laplacian'
                return

        if self.m_proc_img is not None:
            if len(self.m_proc_img.shape) >= 3:
                self.m_proc_img = cv2.cvtColor(self.m_proc_img, cv2.COLOR_BGR2GRAY)

            if self._edgeType_combo_box.currentText() == 'Sobel_XY':
                print("Sobel_XY")
                sobel_img = cv2.Sobel(self.m_proc_img, cv2.CV_8U, 1, 1, ksize=3)
                self.update_image(sobel_img)
            elif self._edgeType_combo_box.currentText() == 'Scharr_X':
                print("Sobel_X")
                s_imageX = cv2.Scharr(self.m_proc_img, cv2.CV_8U, 1, 0)
                self.update_image(s_imageX)
            elif self._edgeType_combo_box.currentText() == 'Scharr_Y':
                print("Sobel_Y")
                s_imageY = cv2.Scharr(self.m_proc_img, cv2.CV_8U, 0, 1)
                self.update_image(s_imageY)
            elif self._edgeType_combo_box.currentText() == 'Laplacian':
                print("Laplacian")
                l_image = cv2.Laplacian(self.m_proc_img, cv2.CV_8U, ksize=3)
                self.update_image(l_image)
            elif self._edgeType_combo_box.currentText() == 'Canny':
                print("Canny")
                c_image1 = cv2.Canny(self.m_proc_img, 150, 300)
                self.update_image(c_image1)
                pyautogui.hotkey('alt', 'tab')

    def update_image(self, img):
        # Creating and scaling QImage
        if len(img.shape) < 3:
            h, w = img.shape
            ch = 1
            img_format = QImage.Format_Grayscale8
        else:
            h, w, ch = img.shape
            img_format = QImage.Format_RGB888
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = QImage(img.data, w, h, ch * w, img_format)
        scaled_img = img.scaled(640, 480, Qt.KeepAspectRatio)
        self.label_image.setFixedSize(scaled_img.width(), scaled_img.height())
        self.label_image.setPixmap(QPixmap.fromImage(scaled_img))

    def load_img_func(self):
        self.kill_thread()
        if self.radiobutton_1.isChecked() is True:
            self.MODE_VIDEO = False
            self.m_main_img = cv2.imread(f"{self.edit.text()}", cv2.IMREAD_COLOR)
            self.m_main_img = cv2.resize(self.m_main_img, (640, 480), interpolation=cv2.INTER_CUBIC)
            self.m_proc_img = self.m_main_img.copy()
            self.update_image(self.m_proc_img)
            print("update image")
        elif self.radiobutton_2.isChecked() is True:
            path = self.edit.text()
            self.createThread_start(path)
        elif self.radiobutton_3.isChecked() is True:
            self.createThread_start(0)

            # self.setup_camera(0)

    def createThread_start(self, vid):
        self.button_haar_start.setEnabled(True)  
        self.button_diff_img.setEnabled(True) 
        self.button_restart.setEnabled(True) 
        self.button_roi.setEnabled(True) 
        self.button_mean_shift.setEnabled(True)

        self.MODE_VIDEO = True
        self.qu = queue.Queue()
        self.qu_img_to_app = queue.Queue()
        self.th_in = project.capture.Thread_in(self.qu)
        self.th_out = project.processing.Thread_out(self.qu, self.qu_img_to_app)

        self.th_in.vid = vid
        self.th_in.start()
        self.th_out.start()
        self.timer.start(15)

    def kill_thread(self):
        self.timer.stop()
        if self.th_in is not None:
            if self.th_in.is_alive() is True:
                self.th_in.status = False
                self.th_in.join()
                print("Thread_in END")
            if self.th_in.capture is not None:
                if self.th_in.capture.isOpened is True:
                    self.th_in.capture.release()

        if self.th_out is not None:
            if self.th_out.is_alive() is True:
                self.th_out.status = False
                self.th_out.join()
                print("Thread_out END")

        if self.labeling_capture is not None:
            self.labeling_capture.release()

        self.button_mean_shift.setEnabled(False) 
        self.button_haar_start.setEnabled(False) 
        self.button_haar_stop.setEnabled(False)  
        self.button_diff_img.setEnabled(False)  
        self.button_restart.setEnabled(False) 
        self.button_roi.setEnabled(False) 
