from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox


class VitParameters(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Cria uma nova janela para definição de parâmetros do tratamento de
        # dados pelo Image Data Generator
        self.setWindowTitle("Configurar Parâmetros da vision transformer")

        self.patch_size = None
        self.projection_dim = None
        self.transformer_layers = None
        self.num_heads = None
        self.mlp_units = None

        # Layout principal, organizando verticalmente os widgets na janela
        layout = QVBoxLayout()

        # Organização dos elementos para tornar possível a entrada de valores
        # para a variável Patch_size
        h_layout = QHBoxLayout() # Layout horizontal para alocar a label da variável e a área de entrada de valores
        h_layout.addWidget(QLabel("Patch size:")) # Adição do widget label ao layout horizontal
        self.patch_size_edit = QLineEdit() # Area de entrada de valores
        self.patch_size_edit.setPlaceholderText("Ex: 16") # Placeholder de texto para o QLineEdit (Servindo como ex.)
        h_layout.addWidget(self.patch_size_edit) # Adição do widget QLineEdit ao layout horizontal
        layout.addLayout(h_layout) # Adição do layout horizonatal deste bloco ao layout vertical principal

        # Projection Dim (Análogo ao bloco de código referente ao Patch_size)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Projection Dim:"))
        self.projection_dim_edit = QLineEdit()
        self.projection_dim_edit.setPlaceholderText("Default: 64")
        h_layout.addWidget(self.projection_dim_edit)
        layout.addLayout(h_layout)

        # Transform Layers (Análogo ao bloco de código referente ao Patch_size)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Transform Layers"))
        self.transformer_layers_edit = QLineEdit()
        self.transformer_layers_edit.setPlaceholderText("Default: 8")
        h_layout.addWidget(self.transformer_layers_edit)
        layout.addLayout(h_layout)

        # Attention Heads (Análogo ao bloco de código referente ao Patch_size)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Attention Heads"))
        self.num_heads_edit = QLineEdit()
        self.num_heads_edit.setPlaceholderText("Default: 4")
        h_layout.addWidget(self.num_heads_edit)
        layout.addLayout(h_layout)

        # Attention Heads (Análogo ao bloco de código referente ao Patch_size)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("MLP Units"))
        self.mlp_units_edit = QLineEdit()
        self.mlp_units_edit.setPlaceholderText("Default: 128")
        h_layout.addWidget(self.mlp_units_edit)
        layout.addLayout(h_layout)

        # Botões OK e Cancel (Análogo à organização do bloco de código referente ao Patch_size)
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

            self.patch_size = int(self.patch_size_edit.text())
            self.projection_dim = int(self.projection_dim_edit.text())
            self.transformer_layers = int(self.transformer_layers_edit.text())
            self.num_heads = int(self.num_heads_edit.text())
            self.mlp_units = int(self.mlp_units_edit.text())

            self.accept()  # fecha o dialog com resultado "aceito"

        except ValueError:
            QMessageBox.information(self, 'Erro', 'Erro ao definir parâmetros, possivelmente algum valor foi inserido '
                                                  'incorretamente. Tente novamente')