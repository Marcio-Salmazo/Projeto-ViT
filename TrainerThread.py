from PyQt5.QtCore import QThread, pyqtSignal
from tensorflow.keras.callbacks import TensorBoard, Callback
from Model import Model
import tensorflow as tf
import PIL

class TrainerThread(QThread):

    log_signal = pyqtSignal(str)  # sinal para enviar mensagens de log ao PyQt
    training_finished = pyqtSignal(bool)

    def __init__(self, neural_network, train_data, val_data, epochs, logName):
        super().__init__()
        self.neural_network = neural_network
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.logName = logName
        self.history = None

    def run(self):

        try:
            model = Model()
            log_path = model.log_directory_manager(self.logName)

            self.log_signal.emit("Iniciando treinamento...")
            self.log_signal.emit(f"Logs armazenados em: {log_path}")

            # callback customizado
            outer = self

            class LogCallback(Callback):
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    msg = (
                        f"Época {epoch + 1}/{self.params['epochs']} - "
                        f"loss: {logs.get('loss', 0):.4f} - "
                        f"acc: {logs.get('accuracy', 0):.4f} - "
                        f"val_loss: {logs.get('val_loss', 0):.4f} - "
                        f"val_acc: {logs.get('val_accuracy', 0):.4f}"
                    )
                    outer.log_signal.emit(msg)

            tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=1)
            log_callback = LogCallback()

            self.neural_network.compile(

                # Define o otimizador adam o qualcombina vantagens do SGD + Momentum + RMSprop.
                # O learning_rate=1e-4 refere-se taxa de aprendizado pequena (0.0001) → útil para estabilizar
                # treinamento de modelos complexos como ViT.
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),

                # Define como função de custo a CategoricalCrossentropy,
                # usada para classificação multiclasse com rótulos one-hot
                # obs: from_logits=True: indica que a última camada do modelo retorna
                # logits (não passou por softmax ainda). Isso permite mais estabilidade numérica.
                # O TensorFlow aplicará softmax internamente antes de calcular a perda.
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

                # Define as métricas que serão monitoradas durante o treinamento e validação.
                # Aqui usamos acurácia, que é a fração de previsões corretas sobre o total.
                metrics=['accuracy']
            )

            self.history = self.neural_network.fit(
                self.train_data,
                epochs=self.epochs,
                validation_data=self.val_data,
                callbacks=[tensorboard_callback, log_callback]
            )

            self.log_signal.emit("Treinamento finalizado com sucesso!")
            self.training_finished.emit(True)

        except Exception as e:
            self.log_signal.emit(f"Erro durante o treinamento: {str(e)}")
            self.training_finished.emit(False)