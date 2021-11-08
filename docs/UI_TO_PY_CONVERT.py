import os
try:
    os.system("python -m PyQt5.uic.pyuic -x MainWindow.ui -o ui_MainWindow.py")
    os.system("python -m PyQt5.uic.pyuic -x TemplateDeleteWindow.ui -o ui_TemplateDeleteWindow.py")
    os.system("python -m PyQt5.uic.pyuic -x TemplateUpdateWindow.ui -o ui_TemplateUpdateWindow.py")

    os.system("Pyrcc5 icons.qrc -o icons_rc.py")

except Exception as e:
    print(e)
    pass


