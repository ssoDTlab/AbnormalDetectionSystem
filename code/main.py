import sys

from PySide6.QtWidgets import QApplication

from project.ui import Window

if __name__ == "__main__":

    app = QApplication()
    w = Window()
    w.show()
    sys.exit(app.exec())
