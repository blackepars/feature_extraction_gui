import glob
import math
import os
import pickle
import time
from datetime import datetime
import imutils
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap, QImage
from skimage.metrics import structural_similarity

from other_functions import *
from natsort import natsorted
from ui_MainWindow import *
from ui_TemplateUpdateWindow import *
from ui_TemplateDeleteWindow import *


class Main_Window(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):

        super(Main_Window, self).__init__(parent=parent)
        if True:
            self.main_window = Ui_MainWindow()
            self.main_window.setupUi(self)

            self.main_window.pushButtonStart.hide()
            self.main_window.lineEditSource.hide()
            self.main_window.labelSourceText.hide()

            self.main_window.labelImageForCrop.installEventFilter(self)

            self.main_window.pushButtonClose_1.clicked.connect(self.close_program)
            self.main_window.pushButtonClose_2.clicked.connect(self.close_program)
            self.main_window.pushButtonClose_3.clicked.connect(self.close_program)

            self.main_window.pushButtonStart.clicked.connect(self.image_timer_start)
            self.main_window.pushButtonTemplateSave.clicked.connect(self.template_save_button)

            self.main_window.pushButtonCurrentTemplatesRefresh.clicked.connect(self.CurrentTemplatesRefresh)
            self.main_window.pushButtonCurrentTemplatesDelete.clicked.connect(self.CurrentTemplatesDeleteButton)

            self.main_window.scrollArea.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.main_window.scrollArea.customContextMenuRequested.connect(self.LastImagesRightClickMenu)

            self.main_window.labelKeypointThreshold.setText(
                str((self.main_window.horizontalSliderKeypointThreshold.value())))
            self.main_window.labelSSIMSliderValue.setText(str((self.main_window.horizontalSliderSSIM.value() / 100)))

            self.main_window.horizontalSliderKeypointThreshold.valueChanged.connect(
                lambda: self.main_window.labelKeypointThreshold.setText(str(
                    (self.main_window.horizontalSliderKeypointThreshold.value()))))

            self.main_window.horizontalSliderSSIM.valueChanged.connect(
                lambda: self.main_window.labelSSIMSliderValue.setText(str(
                    (self.main_window.horizontalSliderSSIM.value() / 100))))

            self.main_window.spinBoxTemplateNumber.valueChanged.connect(self.show_currrent_template)
            self.main_window.comboBoxImageSource.currentTextChanged.connect(lambda text: self.choose_image_source(text))

            self.stylesheet_green = "background-color: rgb(0, 255, 0); border:none; border-radius:15px"
            self.stylesheet_red = "background-color: rgb(255, 0, 0); border:none; border-radius:15px"
            self.stylesheet_yellow = "background-color: rgb(255, 255, 0); border:none; border-radius:15px"

            self.exception_point_old = " "
            self.exception_old = " "

            self.logfile = open("datas/LOG.txt", "a")

            self.last_nok_images_path = "datas/LAST_NOK_IMAGES.pickle"

            self.QUALITY_CONTROL_SITUATION = False
            self.mouse_over_label = False

            self.erode_dilate_value = 3
            self.KeypointThreshold = 40
            self.SSIM_threshold_value = 0.2
            self.blur_value = 5

            self.orb = cv2.ORB_create(nfeatures=1000)
            self.kernel = (7, 7)
            self.show_currrent_template()
            self.image_timer = QtCore.QTimer()
            self.image_read_time = 1000

            self.video_source_type = ""
            self.image_timer.timeout.connect(self.image_read)
            self.image_variable = 1
            self.CurrentTemplatesRefresh()
            self.clear_images()
            self.show_last_nok_images()

    def choose_image_source(self, text):
        try:
            if text == "CHOOSE":
                self.main_window.pushButtonStart.setEnabled(False)
                self.main_window.lineEditSource.hide()
                self.main_window.labelSourceText.hide()

            else:
                self.main_window.pushButtonStart.show()
                self.main_window.pushButtonStart.setEnabled(True)
                self.main_window.lineEditSource.show()
                self.main_window.labelSourceText.show()
                if text == "FOLDER":
                    self.main_window.labelSourceText.setText("FOLDER NAME:")
                    self.main_window.lineEditSource.setText("images")
                if text == "CAMERA":
                    self.main_window.labelSourceText.setText("CAMERA NUMBER:")
                    self.main_window.lineEditSource.setText("0")

                if text == "VIDEO":
                    self.main_window.labelSourceText.setText("VIDEO NAME:")
                    self.main_window.lineEditSource.setText("video.mp4")
        except Exception as e:
            self.log_add("CHOSE VIDEO SOURCE", str(e))
            pass

    def image_read(self):
        try:
            if self.video_source_type == "folder":
                image = cv2.imread(self.image_read_path.format(self.image_variable))
                self.image_variable += 1
                self.image_processing(image)
            elif self.video_source_type == "video":
                ret, image = self.video_source.read()
                self.image_processing(image)

        except Exception as e:
            self.main_window.labelResult.setText(str(e))
            self.main_window.labelResult.setStyleSheet(self.stylesheet_red)
            self.log_add("IMAGE READ", str(e))
            pass

    def image_timer_start(self):
        try:

            if self.main_window.pushButtonStart.text() == "START":
                if not self.image_timer.isActive():

                    if self.main_window.comboBoxImageSource.currentText() == "FOLDER":
                        path = self.main_window.lineEditSource.text()
                        self.image_variable = 1

                        self.image_read_path = path + "/{}.jpg"
                        self.video_source_type = "folder"
                        self.image_read_time = 1000

                    elif self.main_window.comboBoxImageSource.currentText() == "CAMERA":
                        number = int(self.main_window.lineEditSource.text())
                        self.video_source = cv2.VideoCapture(number)
                        self.video_source_type = "video"
                        self.image_read_time = 25

                    elif self.main_window.comboBoxImageSource.currentText() == "VIDEO":
                        video_name = self.main_window.lineEditSource.text()
                        self.video_source = cv2.VideoCapture(video_name)
                        self.video_source_type = "video"
                        self.image_read_time = 25

                    self.image_timer.start(self.image_read_time)
                    self.main_window.pushButtonStart.setText("STOP")

                else:
                    pass
            elif self.main_window.pushButtonStart.text() == "STOP":
                if self.image_timer.isActive():
                    self.image_timer.stop()
                    self.main_window.pushButtonStart.setText("START")
                    self.video_source.release()
                else:
                    pass

        except Exception as e:
            self.log_add("IMAGE TIMER START", str(e))
            pass

    def image_processing(self, current_image):
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        current_image = imutils.resize(current_image, width=640)

        height, width, channel = current_image.shape
        step = channel * width
        self.qImg_for_cut = QImage(current_image.data, width, height, step, QImage.Format_RGB888)
        self.main_window.labelImageForCrop.setPixmap(QPixmap.fromImage(self.qImg_for_cut))

        template = cv2.imread(self.template_image)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

        template = cv2.GaussianBlur(template, (self.blur_value, self.blur_value), 0)

        template_edge = cv2.cvtColor(template.copy(), cv2.COLOR_RGB2GRAY)

        try:
            detect_time_start = time.time()

            for i in range(1):

                self.KeypointThreshold = self.main_window.horizontalSliderKeypointThreshold.value()
                self.main_window.labelKeypointThreshold.setText(str(self.KeypointThreshold))

                self.SSIM_threshold_value = (self.main_window.horizontalSliderSSIM.value()) / 100
                self.main_window.labelSSIMSliderValue.setText(str(self.SSIM_threshold_value))

                current_image = cv2.GaussianBlur(current_image, (self.blur_value, self.blur_value), 0)
                current_image_copy = current_image.copy()

                kp1, des1 = self.orb.detectAndCompute(current_image, None)
                kp2, des2 = self.orb.detectAndCompute(template, None)

                imgKp1 = cv2.drawKeypoints(current_image, kp1, None)
                imgKp2 = cv2.drawKeypoints(template, kp2, None)

                cv2.putText(imgKp1, "{}".format(len(kp1)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 5)
                cv2.putText(imgKp2, "{}".format(len(kp2)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 5)

                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)

                good = []

                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

                MatchCount = len(good)
                self.main_window.labelMatchCount.setText(str(MatchCount))

                if MatchCount >= self.KeypointThreshold:

                    self.main_window.labelTemplateSituation.setStyleSheet(self.stylesheet_green)
                    self.QUALITY_CONTROL_SITUATION = False

                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                    h, w = template.shape[:2]
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                    dst = cv2.perspectiveTransform(pts, M)
                    points = np.int32(dst)
                    points = points.clip(min=0)

                    new_rectangle = np.zeros((4, 2), dtype="int32")
                    i = 0
                    for point in points:
                        new_rectangle[i] = point[0]
                        i += 1

                    cropped = four_point_transform(current_image_copy, new_rectangle)

                    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

                    hegiht, width = template_edge.shape[:2]

                    cropped = cv2.resize(cropped, (width, hegiht))

                    SSIM_Score = structural_similarity(template_edge, cropped, multichannel=True,
                                                       gaussian_weights=True, sigma=1.5,
                                                       use_sample_covariance=False, data_range=1.0)
                    if SSIM_Score < self.SSIM_threshold_value:
                        cropped = cv2.flip(cropped, -1)

                        SSIM_Score_2 = structural_similarity(template_edge, cropped, multichannel=True,
                                                             gaussian_weights=True, sigma=1.5,
                                                             use_sample_covariance=False, data_range=1.0)
                        if SSIM_Score_2 > SSIM_Score:
                            SSIM_Score = SSIM_Score_2
                        else:
                            pass
                    self.main_window.labelSSIMScore.setText("{:.2f}".format(SSIM_Score))

                    if SSIM_Score >= self.SSIM_threshold_value:
                        current_image = cv2.polylines(current_image, [points], True, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.putText(current_image, "OK", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        self.main_window.labelSSIMSituation.setStyleSheet(self.stylesheet_green)

                        h, w, c = current_image.shape
                        zero_image = np.zeros((h, w), dtype="uint8")

                        img3 = cv2.fillPoly(zero_image, [points], (255, 255, 255))

                        current_image_edge = cv2.cvtColor(current_image_copy, cv2.COLOR_BGR2GRAY)

                        current_image_edge = cv2.adaptiveThreshold(current_image_edge, 255,
                                                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                   cv2.THRESH_BINARY_INV, 11, 5)

                        current_image_edge = cv2.erode(current_image_edge, self.kernel, iterations=2)
                        current_image_edge = cv2.dilate(current_image_edge, self.kernel, iterations=1)

                        current_image_edge = cv2.bitwise_and(img3, current_image_edge)

                        contours, hierarchyy = cv2.findContours(image=current_image_edge, mode=cv2.RETR_TREE,
                                                                method=cv2.CHAIN_APPROX_NONE)
                        cv2.drawContours(image=current_image, contours=contours, contourIdx=-1, color=(0, 255, 0),
                                         thickness=2,
                                         lineType=cv2.LINE_AA)

                    else:
                        current_image = cv2.polylines(current_image, [points], True, (255, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(current_image, "NOK", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        self.main_window.labelSSIMSituation.setStyleSheet(self.stylesheet_red)
                        self.QUALITY_CONTROL_SITUATION = True

                    cropped = imutils.resize(cropped, width=int(cropped.shape[1] * 0.8))
                    _, step = cropped.shape
                    self.qImg_EDGE = QImage(cropped, cropped.shape[1], cropped.shape[0], step,
                                            QImage.Format_Grayscale8)

                    template_edge = imutils.resize(template_edge, width=int(template_edge.shape[1] * 0.8))
                    _, step = template_edge.shape
                    self.qImg_EDGE = QImage(template_edge, template_edge.shape[1], template_edge.shape[0],
                                            step,
                                            QImage.Format_Grayscale8)

                else:
                    cv2.putText(current_image, "NOK", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    self.main_window.labelSSIMScore.setText("*")
                    self.main_window.labelTemplateSituation.setStyleSheet(self.stylesheet_red)
                    self.main_window.labelSSIMSituation.setStyleSheet(self.stylesheet_red)
                    self.QUALITY_CONTROL_SITUATION = True

                if self.QUALITY_CONTROL_SITUATION:
                    with open(self.last_nok_images_path, 'rb') as f:
                        IMAGE_KUYRUK = pickle.load(f)
                    if len(IMAGE_KUYRUK) >= 4:
                        del IMAGE_KUYRUK[0]
                    kuyruk_image = cv2.resize(current_image, (260, 160))
                    IMAGE_KUYRUK.append(kuyruk_image)
                    with open(self.last_nok_images_path, 'wb') as f:
                        pickle.dump(IMAGE_KUYRUK, f)
                    self.show_last_nok_images()
                    self.QUALITY_CONTROL_SITUATION = False

                detect_time = (time.time() - detect_time_start) * 1000
                self.main_window.labelDetectTime.setText("{:.2f}".format(detect_time))

                current_image = cv2.resize(current_image, (600, 400))

                height, width, channel = current_image.shape
                step = channel * width
                qImg = QImage(current_image.data, width, height, step, QImage.Format_RGB888)
                self.main_window.labelResult.setPixmap(QtGui.QPixmap.fromImage(qImg)),

        except Exception as e:
            self.log_add("IMAGE PROCESSING", str(e))
            pass

    def show_currrent_template(self):
        try:
            self.template_variable = self.main_window.spinBoxTemplateNumber.value()
            self.template_image = "templates/template_{}.jpg".format(self.template_variable)
            template = cv2.imread(self.template_image)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

            template = cv2.resize(template, (300, 240))
            height, width, channel = template.shape
            step = channel * width
            qImg = QImage(template.data, width, height, step, QImage.Format_RGB888)
            self.main_window.labelCurrentTemplate.setPixmap(QtGui.QPixmap.fromImage(qImg))
        except Exception as e:
            self.main_window.labelCurrentTemplate.setText(str(e))
            self.main_window.labelCurrentTemplate.setStyleSheet(self.stylesheet_red)
            self.log_add("SHOW CURRENT TEMPLATE", str(e))
            pass

    def CurrentTemplatesClear(self):
        try:
            layout = self.main_window.gridLayout_10
            for i in reversed(range(layout.count())):
                layout.itemAt(i).widget().close()
                layout.takeAt(i)
        except Exception as e:
            self.log_add("CURRENT TEMPLATES CLEAR", str(e))
            pass

    def CurrentTemplatesDeleteButton(self):
        degisken = self.main_window.spinBoxTemplateDeleteNumber.text()
        path = "templates/template_{}.jpg".format(degisken)

        recete_olma_durumu = os.path.exists(path)

        if recete_olma_durumu:
            image = cv2.imread(path)
            dialog = Template_Delete_Dialog(image, degisken)

            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                durum, hata = self.CurrentTemplatesDelete(path)
                if durum:
                    self.main_window.labelTemplateDeleteLog.setText("{}. TEMPLATE DELETED".format(degisken))
                else:
                    self.main_window.labelTemplateDeleteLog.setText("TEMPLATE CANNOT DELETED. BECAUSE: {}".format(hata))
                    pass
            else:
                self.main_window.labelTemplateDeleteLog.setText("TEMPLATE DELETE CANCELLED")
                pass
        else:
            self.main_window.labelTemplateDeleteLog.setText("{}. TEMPLATE CANNOT FIND  ".format(degisken))
            pass

    def CurrentTemplatesDelete(self, path):
        try:
            os.remove(path)
            self.CurrentTemplatesRefresh()
            return True, True
        except Exception as e:
            self.log_add("TEMPLATE DELETE", str(e))
            return False, str(e)
            pass

    def CurrentTemplatesRefresh(self):
        self.CurrentTemplatesClear()
        TemplateList = glob.glob1("templates", "*.jpg")
        TemplateList = natsorted(TemplateList)

        TemplateListCount = len(TemplateList)

        TemplateCount = math.ceil(TemplateListCount / 2)
        count_variable = 1
        try:
            for x in range(0, TemplateCount):
                for y in range(0, 3):

                    recete = TemplateList[count_variable]
                    _, TemplateNo = recete.split("_")
                    TemplateNo = TemplateNo.split(".")[0]

                    self.scrollAreaWidgetContents = self.main_window.scrollAreaWidgetContents
                    self.scrollArea = self.main_window.scrollAreaReceteler
                    self.labelTemplate = QtWidgets.QLabel(self.scrollAreaWidgetContents)
                    self.labelTemplate.setMinimumSize(QtCore.QSize(0, 200))
                    self.labelTemplate.setAlignment(QtCore.Qt.AlignCenter)
                    self.labelTemplate.setObjectName("labelTemplate_{}".format(TemplateNo))
                    self.labelTemplate.setText("TEMPLATE {}".format(TemplateNo))

                    image = cv2.imread("templates/template_{}.jpg".format(TemplateNo))
                    image = cv2.resize(image, (300, 200))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    cv2.putText(image, "TEMPLATE {}".format(TemplateNo), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2)
                    height, width, channel = image.shape
                    step = channel * width
                    qImg_for_refresh = QImage(image.data, width, height, step,
                                              QImage.Format_RGB888)
                    self.labelTemplate.setPixmap(QPixmap.fromImage(qImg_for_refresh))

                    self.main_window.gridLayout_10.addWidget(self.labelTemplate, x, y, 1, 1)
                    self.main_window.scrollAreaReceteler.setWidget(self.scrollAreaWidgetContents)
                    self.main_window.gridLayout_4.addWidget(self.scrollArea, 0, 0, 1, 1)
                    count_variable += 1
                    if count_variable >= TemplateListCount:
                        break

                if count_variable >= TemplateListCount:
                    break

        except Exception as e:
            self.log_add("CURRENT TEMPLATES REFRESH", str(e))
            pass

    def show_last_nok_images(self):
        with open(self.last_nok_images_path, 'rb') as f:
            IMAGE_LIST = pickle.load(f)
            IMAGE_LIST.reverse()
        for i in range(0, len(IMAGE_LIST)):

            image = IMAGE_LIST[i]

            height, width, channel = image.shape
            step = channel * width
            self.qImg = QtGui.QImage(image.data, width, height, step,
                                     QtGui.QImage.Format_RGB888)
            if i == 0:
                self.main_window.labelImage0.setPixmap(QtGui.QPixmap.fromImage(self.qImg))
            if i == 1:
                self.main_window.labelImage1.setPixmap(QtGui.QPixmap.fromImage(self.qImg))
            if i == 2:
                self.main_window.labelImage2.setPixmap(QtGui.QPixmap.fromImage(self.qImg))
            if i == 3:
                self.main_window.labelImage3.setPixmap(QtGui.QPixmap.fromImage(self.qImg))

    def clear_images(self):
        self.main_window.labelImage0.clear()
        self.main_window.labelImage1.clear()
        self.main_window.labelImage2.clear()
        self.main_window.labelImage3.clear()
        IMAGE_KUYRUK = []
        with open(self.last_nok_images_path, 'wb') as f:
            pickle.dump(IMAGE_KUYRUK, f)
        self.show_last_nok_images()

    def LastImagesRightClickMenu(self, position):
        # Popup menu
        popMenu = QtWidgets.QMenu()
        popMenu.setStyleSheet("QMenu::item{background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);} "
                              "QMenu::item:selected{background-color: rgb(0, 255, 0);color: rgb(0, 0, 0);} ")
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        popMenu.setFont(font)

        clear_images = QtWidgets.QAction("CLEAR", self)
        popMenu.addAction(clear_images)

        clear_images.triggered.connect(self.clear_images)
        popMenu.exec_(self.main_window.scrollArea.mapToGlobal(position))

    def mousePressEvent(self, eventQMouseEvent):
        if self.mouse_over_label:
            self.originQPoint = eventQMouseEvent.pos()
            self.currentQRubberBand = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
            self.currentQRubberBand.setGeometry(QRect(self.originQPoint, QSize()))
            self.currentQRubberBand.show()
        else:
            pass

    def mouseMoveEvent(self, event_q_mouse_event):
        if self.mouse_over_label:
            try:
                self.currentQRubberBand.setGeometry(
                    QRect(self.originQPoint, event_q_mouse_event.pos()).normalized())
            except Exception as e:
                self.log_add("MOUSE MOVE EXCEPTION ", str(e))
                pass
        else:
            pass

    def mouseReleaseEvent(self, event_q_mouse_event):
        if self.mouse_over_label:
            labelPosition = self.main_window.labelImageForCrop.pos()

            self.currentQRubberBand.hide()
            currentQRect = self.currentQRubberBand.geometry()

            self.currentQRubberBand.deleteLater()

            rectangle = list(currentQRect.getRect())

            try:
                rectangle[0] = (rectangle[0] - labelPosition.x() - 75)
                rectangle[1] = (rectangle[1] - labelPosition.y() - 100)

                self.cropQPixmap = self.qImg_for_cut.copy(
                    QRect(rectangle[0], rectangle[1], rectangle[2], rectangle[3]))

                self.cropQPixmap = self.cropQPixmap.scaled(self.main_window.labelCuttedImage.width(),
                                                           self.main_window.labelCuttedImage.height(),
                                                           Qt.KeepAspectRatio)  # , Qt.SmoothTransformation)

                self.main_window.labelCuttedImage.setPixmap(QPixmap.fromImage(self.cropQPixmap))

            except Exception as e:
                self.log_add("MOUSE RELEASE EXCEPTION:", str(e))
                pass
        else:
            pass

    def eventFilter(self, object, event):
        if event.type() == QEvent.Enter:
            self.mouse_over_label = True
            return True
        elif event.type() == QEvent.Leave:
            self.mouse_over_label = False
        return False

    def template_save_button(self):
        self.template_name_number = self.main_window.spinBoxTemplateName.value()

        path = "templates/template_{}.jpg".format(self.template_name_number)

        is_template_existing_control = os.path.exists(path)
        if is_template_existing_control:
            image1 = cv2.imread(path)
            temporary_path = "templates/temporary_image.jpg"
            self.cropQPixmap.save(temporary_path)
            image2 = cv2.imread(temporary_path)
            os.remove(temporary_path)
            dialog_update = Template_Update_Window(image1, image2, self.template_name_number)
            if dialog_update.exec_() == QtWidgets.QDialog.Accepted:
                self.template_save()
            else:
                self.main_window.labelTemplateAddLog.setText("TEMPLATE SAVE CANCELLED")

        else:
            self.template_save()
            pass

    def template_save(self):
        try:
            self.cropQPixmap.save("templates/template_{}.jpg".format(self.template_name_number))
            self.main_window.labelTemplateAddLog.setText("{}. TEMPLATE SAVED".format(self.template_name_number))
        except Exception as e:
            self.log_add("TEMPLATE ADD:", str(e))
            pass

    def rotating_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def log_add(self, exception_point, exception_name):
        if exception_point != self.exception_point_old and exception_name != self.exception_old:
            datestring = datetime.now().strftime("%d-%m-%Y-%H.%M.%S")
            self.main_window.listWidgetLog.addItem("{} : {} {}".format(datestring, exception_point, exception_name))
            self.logfile.write("{}: {} - {}  \n".format(datestring, str(exception_point), str(exception_name)))
            self.logfile.flush()
        else:
            pass

        self.exception_point_old = exception_point
        self.exception_old = exception_name

    def close_program(self):
        self.log_add("APPLICATION CLOSING", "")
        self.close()


class Template_Delete_Dialog(QtWidgets.QDialog, Ui_TemplateDeleteDialog):
    def __init__(self, image, variable_control, parent=None):
        super(Template_Delete_Dialog, self).__init__(parent)
        self.dialog = Ui_TemplateDeleteDialog()
        self.dialog.setupUi(self)

        self.dialog.pushButtonDelete.clicked.connect(self.template_delete)
        self.dialog.pushButtonCancel.clicked.connect(self.cancel)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (450, 300))
        height, width, channel = image.shape
        step = channel * width
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        self.dialog.labelImage.setPixmap(QPixmap.fromImage(qImg))
        self.dialog.labelAttention.setText("{}. TEMPLATE WILL DELETE !\n ARE YOU SURE ?".format(variable_control))

    def template_delete(self):
        self.accept()

    def cancel(self):
        self.reject()


class Template_Update_Window(QtWidgets.QDialog, Ui_TemplateUpdateDialog):
    def __init__(self, imageCurrent, imageNew, variable_control, parent=None):
        super(Template_Update_Window, self).__init__(parent)
        self.dialog = Ui_TemplateUpdateDialog()
        self.dialog.setupUi(self)

        self.dialog.pushButtonUpdate.clicked.connect(self.template_update)
        self.dialog.pushButtonCancel.clicked.connect(self.cancel)

        imageCurrent = cv2.cvtColor(imageCurrent, cv2.COLOR_BGR2RGB)
        imageCurrent = cv2.resize(imageCurrent, (450, 300))
        height, width, channel = imageCurrent.shape
        step = channel * width
        qImg = QImage(imageCurrent.data, width, height, step, QImage.Format_RGB888)
        self.dialog.labelCurrentTemplate.setPixmap(QPixmap.fromImage(qImg))

        imageNew = cv2.cvtColor(imageNew, cv2.COLOR_BGR2RGB)
        imageNew = cv2.resize(imageNew, (450, 300))
        height, width, channel = imageNew.shape
        step = channel * width
        qImg = QImage(imageNew.data, width, height, step, QImage.Format_RGB888)
        self.dialog.labelNewTemplate.setPixmap(QPixmap.fromImage(qImg))

        self.dialog.labelAttention.setText("{}. TEMPLATE ALREADY EXISTING !\n ARE YOU SURE ?".format(variable_control))

    def template_update(self):
        self.accept()

    def cancel(self):
        self.reject()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    pencere = Main_Window()
    pencere.show()
    sys.exit(app.exec_())
