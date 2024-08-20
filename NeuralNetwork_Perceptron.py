import random
import math

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]  # Pesos
        self.bias = random.uniform(-1, 1)  # Bias

# Função de ativação sigmoide
    def activation(self, x):
        result = 1/(1+math.exp(-x))
        if result > 0.99:
            result = 1
        elif result <= 0.001:
            result = 0
        return result

    def predict(self, x):
        # Calcula a soma ponderada
        summation = self.bias
        for i in range(len(x)):
            
            summation += self.weights[i] * x[i]
        return self.activation(summation)

    def train(self, training_data, labels):
        for epoch in range(self.epochs):
            for i in range(len(training_data)):
                prediction = self.predict(training_data[i])
                error = labels[i] - prediction
                # Atualiza o bias
                self.bias += self.learning_rate * error
                # Atualiza os pesos
                for j in range(len(training_data[i])):
                    self.weights[j] += self.learning_rate * error * training_data[i][j]
                    #print("Peso",self.weights)

# Exemplo de uso
if __name__ == "__main__":
    # Dados de entrada
    training_data = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    # Saídas desejadas
    labels = [0, 0, 0, 1]

    # Cria o perceptron
    perceptron = Perceptron(2)

    # Treina o perceptron
    perceptron.train(training_data, labels)


    # Testa o perceptron
    for x in training_data:
        prediction = perceptron.predict(x)
        print(f"Entrada: {x}, Predicao: {prediction}")
