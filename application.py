# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
import cv2
from utils import detection_utils

detection_graph = None
sess = None

left_right = 20
up_down = 20
btn_height = 40
btn_length = 180

class Ui_MainWIndow(object):

    def setupUi(self, MainWIndow):
        MainWIndow.setObjectName("MainWIndow")
        MainWIndow.resize(2*left_right + 960, 2*up_down + 540 + 2*btn_height)
        self.centralwidget = QtWidgets.QWidget(MainWIndow)
        self.centralwidget.setObjectName("centralwidget")

        # video frame
        self.frame = QtWidgets.QLabel(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(left_right, up_down, 960, 540))
        self.frame.setScaledContents(True)
        self.frame.setObjectName("frame")

        # button to start
        self.btn_start = QtWidgets.QPushButton(self.centralwidget)
        self.btn_start.setGeometry(QtCore.QRect(left_right + btn_length + 20, up_down + 540 + (1/2)*btn_height, btn_length, btn_height))
        self.btn_start.setObjectName("btn_start")
        self.btn_start.clicked.connect(self.on_click_start)
        self.btn_start.setEnabled(False)

        # button to start
        self.btn_load_model = QtWidgets.QPushButton(self.centralwidget)
        self.btn_load_model.setGeometry(QtCore.QRect(left_right, up_down + 540 + (1/2)*btn_height, btn_length, btn_height))
        self.btn_load_model.setObjectName("btn_load_model")
        self.btn_load_model.clicked.connect(self.on_click_load_model)


        # button to stop
        self.btn_stop = QtWidgets.QPushButton(self.centralwidget)
        self.btn_stop.setGeometry(
            QtCore.QRect(2*left_right + 960 - (left_right + btn_length), up_down + 540 + (1 / 2) * btn_height, btn_length, btn_height))
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.clicked.connect(self.on_click_stop)
        self.btn_stop.setEnabled(False)

        MainWIndow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWIndow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 663, 21))
        self.menubar.setObjectName("menubar")
        MainWIndow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWIndow)
        self.statusbar.setObjectName("statusbar")
        self.statusbar.showMessage("Please Load the model")
        MainWIndow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWIndow)
        QtCore.QMetaObject.connectSlotsByName(MainWIndow)

    def retranslateUi(self, MainWIndow):
        _translate = QtCore.QCoreApplication.translate
        MainWIndow.setWindowTitle(_translate("MainWIndow", "Close Cars Detection"))
        self.frame.setText(_translate("MainWIndow", ""))
        self.btn_start.setText(_translate("MainWIndow", "Start"))
        self.btn_load_model.setText(_translate("MainWIndow", "Load Model"))
        self.btn_stop.setText(_translate("MainWIndow", "Stop"))

    def on_click_start(self):
        self.worker1 = Worker1()
        self.worker1.start()
        self.worker1.ImageUpdate.connect(self.image_update_slot)
        self.statusbar.showMessage("STATUS : Detection Ongoing")
        self.btn_stop.setEnabled(True)
        self.btn_start.setEnabled(False)

    def on_click_stop(self):
        _translate = QtCore.QCoreApplication.translate
        self.worker1.stop_running()
        self.statusbar.showMessage("STATUS : Detection Stopped")
        self.btn_start.setText(_translate("MainWIndow", "Start Again"))
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True)

    def on_click_load_model(self):
        self.btn_load_model.setEnabled(False)
        global detection_graph, sess
        _translate = QtCore.QCoreApplication.translate
        detection_graph, sess = detection_utils.load_inference_graph()
        self.statusbar.showMessage("STATUS : Model Loaded successfully")
        self.btn_start.setEnabled(True)

    def image_update_slot(self, image):
        self.frame.setPixmap(QtGui.QPixmap.fromImage(image))


class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    global detection_graph, sess

    def run(self):
        capture = cv2.VideoCapture('data/video2.mp4')
        self.keep_running = True
        while capture.isOpened() and self.keep_running == True:
            ret, frame = capture.read()
            image = frame
            image_height, image_width = image.shape[:2]
            boxes, scores, classes = detection_utils.detect(image, detection_graph, sess)
            image = detection_utils.draw_boxes(image, image_height, image_width, boxes, scores, classes)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_converted = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
            pic = image_converted.scaled(888, 676, QtCore.Qt.KeepAspectRatio)
            self.ImageUpdate.emit(pic)

    def stop_running(self):
        self.keep_running = False


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWIndow = QtWidgets.QMainWindow()
    ui = Ui_MainWIndow()
    ui.setupUi(MainWIndow)
    MainWIndow.show()
    sys.exit(app.exec_())
