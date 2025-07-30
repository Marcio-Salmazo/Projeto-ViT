import os
import PIL
import scipy


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard

import PatchEncoder
import PatchExtractor
from PatchEncoder import PatchEncoder
from PatchExtractor import PatchExtractor


# A classe tem como objetivo construir a arquitetura da rede, bem como
# permitir a etapa de compilação e treinamento
class ModelCreator:

    def __init__(self, input_shape=None, patch_size=None, num_patches=None, projection_dim=None,
                 transformer_layers=None, num_heads=None, mlp_units=None, num_classes=None):
        # Parâmetros para o ViT
        # -------------------------------------------------------------------------

        self.input_shape = input_shape  # formato da imagem de entrada (ex: (224, 224, 3))
        self.patch_size = patch_size  # tamanho dos blocos em que a imagem será dividida (ex: 16)
        self.num_patches = num_patches  # total de patches da imagem (ex: 196)
        self.projection_dim = projection_dim  # dimensão dos vetores em que os patches serão projetados (ex: 512)
        self.transformer_layers = transformer_layers  # número de blocos do encoder Transformer (ex: 8)
        self.num_heads = num_heads  # número de cabeças de atenção (multi-head attention)
        self.mlp_units = mlp_units  # número de neurônios da MLP interna (ex: 128)
        self.num_classes = num_classes  # total de classes do problema de classificação

    # Construção da arquitetura do ViT
    # Recebe como parâmetros os subsets de treino e validação
    def vit_classifier(self):
        # Aqui é definida a entrada do modelo, criando um placeholder para imagens com a
        # forma especificada por self.input_shape
        inputs = layers.Input(shape=self.input_shape)
        # Usa a camada personalizada PatchExtractor para dividir a imagem em patches não sobrepostos.
        patches = PatchExtractor(self.patch_size)(inputs)
        # Aplica a camada PatchEncoder, responsável por projetar cada patch em um vetor projection_dim
        # além de adiciona codificação posicional
        encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(patches)

        # Blocos Transformer: executa vários blocos de Transformer Encoder (quantidade definida por transformer_layers)
        for _ in range(self.transformer_layers):
            # Aqui ocorre o processo de normalização por camada, antes do processo de atenção
            # Importante salientar que epsilon = 1e-6 representa um pequeno valor para evitar divisão por zero.
            # Tal processo é importante por melhorar a estabilidade do treinamento
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

            # Aplica atenção multi-cabeça (multi-head self-attention), onde cada "cabeça" foca em partes
            # diferentes da sequência de patches. x1 é usado como query, key e value (self-attention) e
            # dropout=0.1: evita overfitting.
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)

            # Aplica uma conexão residual. Soma a saída da atenção com a entrada original, preservando
            # a informação inicial e melhora o fluxo de gradiente.
            x2 = layers.Add()([attention_output, encoded_patches])

            # Normaliza novamente a sequência, antes de passar por uma MLP.
            # A normalização antes de cada sub-bloco é prática padrão em Transformers.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

            # Aplica um MLP com duas camadas:
            # Primeira camada: expande a capacidade (ex: 512 → 2048)
            # Segunda camada: retorna ao projection_dim (ex: 2048 → 512)
            # A ativação GELU é suave e eficaz para Transformers.
            x3 = layers.Dense(self.mlp_units, activation=tf.nn.gelu)(x3)
            x3 = layers.Dense(self.projection_dim)(x3)

            # Outra conexão residual, agora para o sub-bloco MLP.
            # Resultado é a entrada para o próximo bloco Transformer, se houver mais
            encoded_patches = layers.Add()([x3, x2])

        # Trecho referente à cabeça de classificação
        # Normaliza o vetor final de cada patch após os blocos do encoder.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # "Achata" todos os vetores em um único vetor plano por imagem.
        # Isso cria um vetor representando toda a imagem (todos os patches).
        representation = layers.Flatten()(representation)

        # Aplica dropout com 50% de taxa, ajudando a reduzir o overfitting.
        representation = layers.Dropout(0.5)(representation)

        # Aplica uma MLP final para extrair boas características antes da classificação.
        # Outro dropout é aplicado.
        features = layers.Dense(self.mlp_units, activation=tf.nn.gelu)(representation)
        features = layers.Dropout(0.5)(features)

        # Na última camada, temos:
        # Gera um vetor com num_classes valores (logits).
        # Cada valor representa o grau de associação com uma classe.
        logits = layers.Dense(self.num_classes)(features)

        # Cria o modelo final com entrada inputs e saída logits.
        model = tf.keras.Model(inputs=inputs, outputs=logits)

        # Realiza a chamada da função para compilar e treinar o modelo
        # self.vit_compile_train(model, train_generator, validation_generator, eps)
        return model

    def vit_compile_train(self, model, train_generator, validation_generator, eps):
        # Esse metodo prepara o modelo para o treinamento.
        model.compile(

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

        # Criar o diretório "logs/fit/" caso não exista
        log_dir = "logs/fit/"
        os.makedirs(log_dir, exist_ok=True)  # Garante que o diretório existe

        # Criar um subdiretório único para cada execução
        run_id = '_run_' + str(len(os.listdir(log_dir)) + 1)
        log_dir = os.path.join(log_dir, run_id)

        # Definir o callback do TensorBoard
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Esse é o metodo que executa o treinamento supervisionado do modelo.
        # Ele é responsável por:
        #
        #   a - Passar os dados de entrada pelo modelo (forward pass)
        #   b - Calcular a perda
        #   c - Propagar os erros para trás (backpropagation)
        #   d - Atualizar os pesos com base nos gradientes
        model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=eps,
            callbacks=[tensorboard_callback]
        )

        # Salvar os pesos após o treinamento
        model.save_weights('vit_eyes.weights.h5')
