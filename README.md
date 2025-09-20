# 📸 Rede Neural com Vision Transformer

Este projeto tem como objetivo implementar a arquitetura Vision Transformer (ViT) de forma acessível e intuitiva, utilizando uma interface gráfica via PyQt5 que facilita a exploração e aplicação dessa tecnologia em diferentes cenários.O Vision Transformer representa uma mudança de paradigma em visão computacional, pois adapta os mecanismos de atenção originalmente desenvolvidos para processamento de linguagem natural (os Transformers) ao domínio de imagens. Em vez de processar uma imagem por meio de convoluções, o ViT a divide em pequenos blocos (patches), que são tratados como "palavras visuais". Esses blocos são então passados por camadas de autoatenção, que permitem ao modelo aprender relações globais entre diferentes regiões da imagem desde as primeiras etapas do processamento.

Entre suas principais vantagens, destacam-se a capacidade de captar dependências de longo alcance sem a limitação de janelas locais das convoluções, além de uma grande escalabilidade: com mais dados e maior poder computacional, os Vision Transformers tendem a superar modelos tradicionais de redes convolucionais em diversas tarefas, como classificação de imagens, segmentação semântica e detecção de objetos. Esta aplicação serve como um dos módulos para meu projeto de mestrado.

---

## 🔎 Descrição Geral

O **sistema** tem como objetivo:
- Preparar dados para treinamento (Separando os grupos de treinamento e teste);
- Construir e compilar a estrutura da rede neural (aplicando os parâmetros necessários para seu funcionamento);
- Treinar a rede neural e devolver resultados via logs para o Tensorboard, bem como armazenar o arquivo .w5 contendo os pesos aprendidos;

Destina-se principalmente a **pesquisadores na área de ciencia de dados**.

---

## 🖥️ Interface e Funcionalidades

### 📂 Janela Inicial
- Contém uma área à esquerda dedicada à exibição das mensagens de log (informando sobre o status da operação);
- Contém uma área à direita dedicada às funcionalidades do sistema, sendo:
  - **Selecionar dataset** – Permite selecionar a pasta contendo a base de dados para o treinamento. Importante salientar que o diretório escolhido deve conter subpastas (cada uma representando as diferentes classes). Essa função exige a definição do tamanho de entrada, tamanho dos lotes (batch) e porcentagem de divisão para os dados de validação;
  - **Construir Modelo ViT** – Constrói e compila a arquiteura da rede. Essa função exige a definição de parâmetros específicos à ViT, os quais são detalhados na seção 'Parâmetros exigidos pelo programa' deste mesmo documento  
  - **Iniciar treinamento** – Inicia o treinamento da rede. Para ter início, exige a definição de um nome para o arquivo de log e a quantidade de épocas para o treinamento;  
  - **Abrir TensorBoard** – Inicia o Tensorboard e abre uma página na web para exibição dos arquivos de log. Esta função exige a escolha do diretório que contém os logs (geralmente está em logs/fit na pasta raiz do executável);  
  - **Fechar programa** – encerra a aplicação. 

---

### ✂️ Parâmetros exigidos pelo programa
- **Input size** - Tamanho que as imagens devem ser redimensionadas para servir como entrada da rede. O valor inserido definira a altura e largura da imagem;
- **Batch size** - Refere-se ao número de amostras de dados que um modelo de aprendizado de máquina processa em uma única iteração;
- **Split (treino/validação)** - Define a porcentagem de dados destinados para treino e validação. Exemplo: 0.2 -> 20% para validação e 80% para treino.

- **Patch size** - O tamanho dos blocos (patches) em que a imagem será dividida. Quanto menor o patch, mais detalhes o modelo enxerga desde o início, mas também aumenta a quantidade de patches a processar (mais custo computacional);
- **Projection Dim** - A dimensão do vetor em que cada patch será representado após a projeção linear (Dimensões maiores permitem mais capacidade de representação, mas também exigem mais memória e poder de processamento);
- **Transform Layers** - Número de blocos de transformers (compostos por atenção + MLP) empilhados no modelo. Quanto mais camadas, mais refinada e abstrata fica a representação;
- **Attention Heads** - Cada camada de atenção pode ter várias "cabeças", que aprendem a focar em diferentes aspectos da imagem ao mesmo tempo;
- **MLP Units** - Número de neurônios nas camadas densas (feed-forward layers) que seguem a parte de atenção em cada bloco do transformador. Normalmente é um valor maior que o 'projection dim'.

---

> 🔎 **Observações Importantes**  
> - O valor de 'Split' deve estar em notação de ponto flutuante, estritamente entre 0.0 e 1.0;
> - O aplicativo indica valores 'padrões' caso o usuário não saiba ao certo o valor de alguns parâmetros;
> - O diretório escolhido para o dataset deve conter subpastas (cada uma representando as diferentes classes);  
> - Seguir as versões dos pré-requisitos à risca, uma vez que versões mais novas podem gerar conflitos na IDE.


## ⚙️ Pré-requisitos e Instalação

- Sistema Operacional: **Windows**  
- Python **3.9** (recomendado)  
- Tensorflow 2.10.0
- Numpy 1.23.5
- Scipy 1.13.1
- Protobuf 3.20.2
- Tensorboard 2.10.1

---

## ▶️ Modo de Uso

1. Abrir uma IDE python de sua preferência e criar um novo ambiente virtual
2. Instalar os pacotes utilizados pela aplicação
3. Executar o arquivo main.py
4. Carregar um diretório contendo uma base de dados e definir os parâmetros exigidos
5. Construir a estrutura da arquitetura
6. Iniciar o treinamento, definindo os parâmetros exigidos
7. Aguardar até o encerramento do treino para obter o arquivo de pesos e logs

---

## ⚠️ Erros Comuns

| Erro | Causa provável | Solução |
|------|----------------|---------|
| ❌ Erro ao abrir base de dados | Estrutura de arquivos inválida | Verifique se o diretório contém as subpastas como categorias |
| ❌ Aplicativo não abre | Python ou dependências ausentes | Reinstale dependências |
| ❌ Travamento ou fechamento inesperado | Instabilidade de código | Contatar desenvolvedor |

---

## 🆕 Atualizações / Changelog

- **v1.0.0-beta**
  - Inclusão da interface gráfica com PyQt5
  - Inclusão de elementos gráficos
  - Inclusão do direcionamento para o TensorBoard
  
---

## 👨‍💻 Autores / Contribuidores

- Marcio Salmazo Ramos – **Desenvolvedor principal**  
  📧 marcio.salmazo19@gmail.com  
- Daniel Duarte Abdala  

---
