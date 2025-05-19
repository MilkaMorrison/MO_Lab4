from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QLabel, QTextEdit, QGroupBox, QComboBox,
                             QHBoxLayout, QPushButton, QLineEdit)
from Penalty_method_interface import Penalty_method
from Barrier_function_method_interface import Barrier_function_method

class InterfaceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Методы условной оптимизации")
        self.resize(500, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_panel = QWidget()
        #left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)


        func_group = QGroupBox("Параметры функции")
        func_layout = QVBoxLayout()
        self.f_input = self.create_input("Коэффициенты функции f:")
        self.g_input = self.create_input("Коэффициенты функции ограничений g:")
        func_layout.addWidget(self.f_input)
        func_layout.addWidget(self.g_input)
        func_group.setLayout(func_layout)

        algo_group = QGroupBox("Параметры алгоритма")
        algo_layout = QVBoxLayout()
        self.x_input = self.create_input("Начальная точка x:")
        self.r_input = self.create_input("Параметр штрафа r:")
        self.C_input = self.create_input("Число C:")
        self.e_input = self.create_input("Точность e:")
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Метод штрафов",
            "Метод барьерных функций"
        ])

        algo_layout.addWidget(self.x_input)
        algo_layout.addWidget(self.r_input)
        algo_layout.addWidget(self.C_input)
        algo_layout.addWidget(self.e_input)
        method_label = QLabel("Выберите метод:")
        algo_layout.addWidget(method_label)
        algo_layout.addWidget(self.method_combo)
        algo_group.setLayout(algo_layout)

        self.calc_button = QPushButton("Рассчитать")
        self.calc_button.clicked.connect(self.run_calculation)

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        left_layout.addWidget(func_group)
        left_layout.addWidget(algo_group)
        left_layout.addWidget(self.calc_button)
        left_layout.addWidget(self.output)

        main_layout.addWidget(left_panel)

    def create_input(self, label_text, default=""):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        label = QLabel(label_text)
        input_field = QLineEdit(default)
        layout.addWidget(label)
        layout.addWidget(input_field)
        return widget

    def run_calculation(self):
        try:
            f_coef = list(map(float, self.f_input.findChild(QLineEdit).text().split()))
            g_coef = list(map(float, self.g_input.findChild(QLineEdit).text().split()))
            x = list(map(float, self.x_input.findChild(QLineEdit).text().split()))
            e = float(self.e_input.findChild(QLineEdit).text())
            r = float(self.r_input.findChild(QLineEdit).text())
            C = float(self.C_input.findChild(QLineEdit).text())
            method = self.method_combo.currentText()

            if method == "Метод штрафов":
                result, points = Penalty_method(f_coef, g_coef, r, C, e, x)
            else:
                result, points = Barrier_function_method(f_coef, g_coef, r, C, e, x)

            self.output.clear()
            self.output.append(f"Найденная точка: ({result['point'][0]:.3f}, {result['point'][1]:.3f})")
            self.output.append(f"Значение функции: {result['value']:.3f}")
            self.output.append(f"Количество итераций: {result['iterations']}")

        except Exception as e:
            self.output.append(f"Ошибка: {str(e)}")


if __name__ == "__main__":
    app = QApplication([])
    window = InterfaceApp()
    window.show()
    app.exec()