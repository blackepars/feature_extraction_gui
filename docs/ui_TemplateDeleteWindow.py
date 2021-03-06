# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'TemplateDeleteWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TemplateDeleteDialog(object):
    def setupUi(self, TemplateDeleteDialog):
        TemplateDeleteDialog.setObjectName("TemplateDeleteDialog")
        TemplateDeleteDialog.resize(468, 432)
        self.verticalLayout = QtWidgets.QVBoxLayout(TemplateDeleteDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.labelImage = QtWidgets.QLabel(TemplateDeleteDialog)
        self.labelImage.setMinimumSize(QtCore.QSize(450, 300))
        self.labelImage.setText("")
        self.labelImage.setAlignment(QtCore.Qt.AlignCenter)
        self.labelImage.setObjectName("labelImage")
        self.verticalLayout.addWidget(self.labelImage)
        self.labelAttention = QtWidgets.QLabel(TemplateDeleteDialog)
        self.labelAttention.setMinimumSize(QtCore.QSize(0, 50))
        self.labelAttention.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.labelAttention.setFont(font)
        self.labelAttention.setStyleSheet("background-color: rgb(85, 170, 255);\n"
"color: rgb(255, 0, 0);\n"
"border: 2px solid;\n"
"border-radius: 10px;\n"
"")
        self.labelAttention.setAlignment(QtCore.Qt.AlignCenter)
        self.labelAttention.setObjectName("labelAttention")
        self.verticalLayout.addWidget(self.labelAttention)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButtonDelete = QtWidgets.QPushButton(TemplateDeleteDialog)
        self.pushButtonDelete.setMinimumSize(QtCore.QSize(150, 50))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pushButtonDelete.setFont(font)
        self.pushButtonDelete.setStyleSheet("QPushButton{\n"
"    background-color: rgb(85, 170, 255);\n"
"    qproperty-icon: url(:/images/icons/delete.png);\n"
"    color: rgb(0, 0, 0);\n"
"    border: 5px solid;\n"
"    border-radius: 10px;\n"
"}\n"
"QPushButton::hover\n"
"{\n"
"    background-color: rgb(255, 0, 0);\n"
"\n"
"}")
        self.pushButtonDelete.setIconSize(QtCore.QSize(30, 30))
        self.pushButtonDelete.setObjectName("pushButtonDelete")
        self.horizontalLayout.addWidget(self.pushButtonDelete)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.pushButtonCancel = QtWidgets.QPushButton(TemplateDeleteDialog)
        self.pushButtonCancel.setMinimumSize(QtCore.QSize(150, 50))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pushButtonCancel.setFont(font)
        self.pushButtonCancel.setStyleSheet("QPushButton{\n"
"    background-color: rgb(85, 170, 255);\n"
"    qproperty-icon: url(:/images/icons/close_icon.png);\n"
"    color: rgb(0, 0, 0);\n"
"    border: 5px solid;\n"
"    border-radius: 10px;\n"
"}\n"
"QPushButton::hover\n"
"{\n"
"    background-color: rgb(0, 255, 0);\n"
"\n"
"}")
        self.pushButtonCancel.setIconSize(QtCore.QSize(30, 30))
        self.pushButtonCancel.setObjectName("pushButtonCancel")
        self.horizontalLayout.addWidget(self.pushButtonCancel)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(TemplateDeleteDialog)
        QtCore.QMetaObject.connectSlotsByName(TemplateDeleteDialog)

    def retranslateUi(self, TemplateDeleteDialog):
        _translate = QtCore.QCoreApplication.translate
        TemplateDeleteDialog.setWindowTitle(_translate("TemplateDeleteDialog", "TEMPLATE DELETE"))
        self.labelAttention.setText(_translate("TemplateDeleteDialog", "*"))
        self.pushButtonDelete.setText(_translate("TemplateDeleteDialog", "DELETE"))
        self.pushButtonCancel.setText(_translate("TemplateDeleteDialog", "CANCEL"))
import icons_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    TemplateDeleteDialog = QtWidgets.QDialog()
    ui = Ui_TemplateDeleteDialog()
    ui.setupUi(TemplateDeleteDialog)
    TemplateDeleteDialog.show()
    sys.exit(app.exec_())
