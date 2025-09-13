from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox


class DataParameters(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Cria uma nova janela para definição de parâmetros do tratamento de
        # dados pelo Image Data Generator
        self.setWindowTitle("Configurar Parâmetros de Dataset")

        self.input_size = None
        self.batch_size = None
        self.split = None

        # Layout principal, organizando verticalmente os widgets na janela
        layout = QVBoxLayout()

        # Organização dos elementos para tornar possível a entrada de valores
        # para a variável Input_Size
        h_layout = QHBoxLayout() # Layout horizontal para alocar a label da variável e a área de entrada de valores
        h_layout.addWidget(QLabel("Input Size:")) # Adição do widget label ao layout horizontal
        self.input_size_edit = QLineEdit() # Area de entrada de valores
        self.input_size_edit.setPlaceholderText("Ex: 224") # Placeholder de texto para o QLineEdit (Servindo como ex.)
        h_layout.addWidget(self.input_size_edit) # Adição do widget QLineEdit ao layout horizontal
        layout.addLayout(h_layout) # Adição do layout horizonatal deste bloco ao layout vertical principal

        # Batch Size (Análogo ao bloco de código referente ao Input_Size)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_edit = QLineEdit()
        self.batch_size_edit.setPlaceholderText("Ex: 32")
        h_layout.addWidget(self.batch_size_edit)
        layout.addLayout(h_layout)

        # Split (treino/validação) (Análogo ao bloco de código referente ao Input_Size)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Split (treino/validação):"))
        self.split_edit = QLineEdit()
        self.split_edit.setPlaceholderText("Ex: 0.8")
        h_layout.addWidget(self.split_edit)
        layout.addLayout(h_layout)

        # Botões OK e Cancel (Análogo à organização do bloco de código referente ao Input_Size)
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancelar")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Conectar os botões à funcionalidades do sistema
        ok_button.clicked.connect(self.accept_data)
        cancel_button.clicked.connect(self.reject)

    def accept_data(self):
        try:
            self.input_size = int(self.input_size_edit.text())
            self.batch_size = int(self.batch_size_edit.text())
            self.split = float(self.split_edit.text())

            # valida o intervalo do split
            if not (0 < self.split < 1):
                QMessageBox.warning(self, "Erro de valor", "O split deve ser um número entre 0 e 1 (ex: 0.8).")
                return

            self.accept()  # fecha o dialog com resultado "aceito"
        except ValueError:
            QMessageBox.information(self, 'Erro', 'Erro ao definir parâmetros, possivelmente algum valor foi inserido '
                                                  'incorretamente. Tente novamente')