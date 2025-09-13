from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox


class NetworkLogName(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Cria uma nova janela para definição de parâmetros do tratamento de
        # dados pelo Image Data Generator
        self.setWindowTitle("Nome de arquivo de LOG (Tensorboard)")

        self.log_name = None
        self.epochs = None

        # Layout principal, organizando verticalmente os widgets na janela
        layout = QVBoxLayout()

        # Organização dos elementos para tornar possível a entrada de valores
        h_layout = QHBoxLayout() # Layout horizontal para alocar a label da variável e a área de entrada de valores
        h_layout.addWidget(QLabel("Escolha um nome para o arquivo de LOG (Tensorboard):")) # Adição do widget label ao layout horizontal
        self.name_edit = QLineEdit() # Area de entrada de valores
        self.name_edit.setPlaceholderText("Ex: name_100epochs_split0.3") # Placeholder de texto para o QLineEdit (Servindo como ex.)
        h_layout.addWidget(self.name_edit) # Adição do widget QLineEdit ao layout horizontal
        layout.addLayout(h_layout) # Adição do layout horizonatal deste bloco ao layout vertical principal

        # Organização dos elementos para tornar possível a entrada de valores
        h_layout = QHBoxLayout()  # Layout horizontal para alocar a label da variável e a área de entrada de valores
        h_layout.addWidget(QLabel(
            "Escolha a quantidade de épocas de treinamento:"))  # Adição do widget label ao layout horizontal
        self.epochs_edit = QLineEdit()  # Area de entrada de valores
        self.epochs_edit.setPlaceholderText('250')  # Placeholder de texto para o QLineEdit (Servindo como ex.)
        h_layout.addWidget(self.epochs_edit)  # Adição do widget QLineEdit ao layout horizontal
        layout.addLayout(h_layout)  # Adição do layout horizonatal deste bloco ao layout vertical principal

        # Botões OK e Cancel (Análogo à organização do bloco de código anterior)
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
            self.log_name = str(self.name_edit.text())
            self.epochs = int(self.epochs_edit.text())

            # valida se o valor das épocas é inteiro positivo
            if not self.epochs > 0:
                QMessageBox.warning(self, "Erro", "O valor da quantidade de épocas deve ser positivo e inteiro")
                return

            self.accept()  # fecha o dialog com resultado "aceito"
        except ValueError:
            QMessageBox.information(self, 'Erro', 'Erro ao definir parâmetros, possivelmente algum valor foi inserido '
                                                  'incorretamente. Tente novamente')