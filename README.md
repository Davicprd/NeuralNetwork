# Redes Neurais Convolucionais

Neste repositório, foram desenvolvidas Redes Neurais Convolucionais (CNN) utilizando TensorFlow e Keras para a tarefa de classificação de imagens, com foco no reconhecimento de diferentes categorias a partir de um dataset de imagens (Cães e Gatos / Flores).

**Tecnologias e bibliotecas utilizadas:**

- Python
- TensorFlow / Keras (para criação, treinamento e avaliação do modelo)
- pathlib (para manipulação de caminhos de diretórios)
- Dataset local (pasta contendo imagens organizadas em subdiretórios por classe)

# Etapas do Projeto:

**Carregamento e Pré-processamento dos Dados**

- Utilização da função `image_dataset_from_directory` do Keras para carregar as imagens do diretório local, dividindo automaticamente o conjunto em dados de treinamento (80%) e validação (20%).

- Definição de parâmetros como tamanho das imagens e batch size.

- Aplicação de cache e prefetching com `tf.data.AUTOTUNE` para otimizar o desempenho no treinamento.

**Aumento de Dados (Data Augmentation)**

Implementação de um pipeline de aumento de dados com transformações aleatórias como, Espelhamento horizontal, Pequenas rotações, Zoom aleatório, Isso ajuda a melhorar a generalização do modelo e evita overfitting.

**Construção das Reded Neurais Convolucionais (CNN)**

As arquiteturas foram construídas utilizando modelos Sequential do Keras, compostos por blocos convolucionais, contendo camadas Conv2D com ReLU, BatchNormalization, MaxPooling2D, Dropout. Após os blocos convolucionais, as redes contam com uma Camada Flatten para vetorização dos dados e camadas densas (Dense), sendo a última com ativação softmax para saída multiclasses.

**Compilação e Treinamento**

- Otimizador: Adam
- Função de perda: Sparse Categorical Crossentropy
- Métrica de desempenho: Acurácia



