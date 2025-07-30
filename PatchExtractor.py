import tensorflow as tf
from tensorflow.keras import layers


# A classe é responsável por definir os patches
# que serve para dividir a imagem em pequenos blocos.
# Importante salientar que a classe herda de
# tf.keras.layers.Layer, com isso, torna-se possível
# construir uma camada personalizada, com lógica específica.
class PatchExtractor(layers.Layer):

    # Contrutor da classe
    def __init__(self, patch_size):
        # super().__init__() chama o construtor da classe mãe (Layer),
        # garantindo a inicialização correta da camada no TensorFlow.
        super(PatchExtractor, self).__init__()
        # O patch_size define as dimensões de cada patch
        self.patch_size = patch_size

    # O metodo call define o que a camada faz quando ela é chamada durante o modelo
    # É necessário compreender que a classe está sendo usada como se fosse uma função chamável.
    # Em Python, isso é chamado de overloading do operador (), e é habilitado por meio do
    # No Keras, __call__ é implementado na superclasse Layer, e ele chama internamente
    # o call(...) que foi definido abaixo define.

    # obs: Toda camada personalizada que herda de tf.keras.layers.Layer deve:
    #   1 - Definir a lógica de computação no call(...)
    #   2 - Usar o __init__ apenas para armazenar hiperparâmetros e instanciar subcamadas

    # obs2: O parâmetro images é o tensor de entrada, ou seja, a representação numérica em forma
    # de tensor de uma imagem  (ou de qualquer outro dado), passada para a rede neural, com
    # shape típico (batch_size, height, width, channels)

    def call(self, images):
        # Obtém dinamicamente o batch_size, ou seja, quantas imagens estão no lote atual.
        # Isso é necessário para reconstruir os patches corretamente mais tarde.
        batch_size = tf.shape(images)[0]

        # Aqui é feita a operação central que divide cada imagem em pequenos patches não sobrepostos.
        # Tal operação é feita pela função tf.image.extract_patches(...)
        patches = tf.image.extract_patches(

            # Tensor de entrada
            images=images,
            # 'Sizes' define a dimensão dos trechos extraídos em cada imagem (dimensões dos patches)
            # As extremidades recebem o valor 1 pois não estamos dividindo lotes ou canais da imagem
            sizes=[1, self.patch_size, self.patch_size, 1],
            # 'Strides' define o passo do 'deslizamento' do corte dos patches, ele deve ser configurado
            # com os mesmos valores definidos em 'sizes' (dimensão dos patches) para que NÃO haja sobreposição
            strides=[1, self.patch_size, self.patch_size, 1],
            # 'Rates' define uma dilatação, todos os valores definidos como 1 indica que NÃO deve haver dilatação
            rates=[1, 1, 1, 1],
            # 'Padding' configura preenchimento extra nas bordas
            # neste caso, pode descartar partes se não couberem perfeitamente
            padding='VALID'

        )

        # Obtém o número de elementos em cada patch  pois os patches são achatados (flattened)
        # automaticamente pela função extract_patches. (ex: sendo um patch com as dimensões 16x16,
        # cada patch contém 768 elementos, visto que 16*16*3 = 768)
        patch_dims = tf.shape(patches)[-1]
        num_patches = tf.shape(patches)[1] * tf.shape(patches)[2]

        # O tensor é reformatado para shape: (batch_size, num_patches, patch_dims), com isso
        # temos algo que se parece com uma sequência de vetores — pronto para ser tratado
        # como uma entrada para o Transformer.

        # obs: num_patches = num_patches_y * num_patches_x → ex: 196
        # obs2: patch_dims = tamanho do vetor flatten de cada patch → ex: 768
        patches = tf.reshape(patches, [batch_size, num_patches, patch_dims])

        # Retorna os patches
        return patches
