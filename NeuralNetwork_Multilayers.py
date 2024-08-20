import random
import math

# Função de ativação sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivada da função sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Classe para a Rede Neural
class NeuralNetwork:
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        # Inicializa os pesos com valores aleatórios
        self.weights_ih = [[random.uniform(-1, 1) for _ in range(hidden_neurons)] for _ in range(input_neurons)]
        self.weights_ho = [[random.uniform(-1, 1) for _ in range(output_neurons)] for _ in range(hidden_neurons)]
        # Inicializa os bias com valores aleatórios
        self.bias_h = [random.uniform(-1, 1) for _ in range(hidden_neurons)]
        self.bias_o = [random.uniform(-1, 1) for _ in range(output_neurons)]

    def feedforward(self, inputs):
        # Calcula os valores para a camada escondida
        hidden = [0] * self.hidden_neurons
        for h in range(self.hidden_neurons):
            for i in range(self.input_neurons):
                hidden[h] += inputs[i] * self.weights_ih[i][h]
            hidden[h] += self.bias_h[h]
            hidden[h] = sigmoid(hidden[h])

        # Calcula os valores para a camada de saída
        outputs = [0] * self.output_neurons
        for o in range(self.output_neurons):
            for h in range(self.hidden_neurons):
                outputs[o] += hidden[h] * self.weights_ho[h][o]
            outputs[o] += self.bias_o[o]
            outputs[o] = sigmoid(outputs[o])

        return outputs

    def train(self, inputs, targets, learning_rate):
        # Feedforward
        hidden = [0] * self.hidden_neurons
        for h in range(self.hidden_neurons):
            for i in range(self.input_neurons):
                hidden[h] += inputs[i] * self.weights_ih[i][h]
            hidden[h] += self.bias_h[h]
            hidden[h] = sigmoid(hidden[h])

        outputs = [0] * self.output_neurons
        for o in range(self.output_neurons):
            for h in range(self.hidden_neurons):
                outputs[o] += hidden[h] * self.weights_ho[h][o]
            outputs[o] += self.bias_o[o]
            outputs[o] = sigmoid(outputs[o])

        # Calcula o erro das saídas
        output_errors = [0] * self.output_neurons
        for o in range(self.output_neurons):
            output_errors[o] = targets[o] - outputs[o]

        # Calcula os deltas para a camada de saída
        output_deltas = [0] * self.output_neurons
        for o in range(self.output_neurons):
            output_deltas[o] = output_errors[o] * sigmoid_derivative(outputs[o])

        # Calcula o erro da camada escondida
        hidden_errors = [0] * self.hidden_neurons
        for h in range(self.hidden_neurons):
            error = 0
            for o in range(self.output_neurons):
                error += output_deltas[o] * self.weights_ho[h][o]
            hidden_errors[h] = error

        # Calcula os deltas para a camada escondida
        hidden_deltas = [0] * self.hidden_neurons
        for h in range(self.hidden_neurons):
            hidden_deltas[h] = hidden_errors[h] * sigmoid_derivative(hidden[h])

        # Atualiza os pesos e bias da camada de saída
        for o in range(self.output_neurons):
            for h in range(self.hidden_neurons):
                self.weights_ho[h][o] += learning_rate * output_deltas[o] * hidden[h]
            self.bias_o[o] += learning_rate * output_deltas[o]

        # Atualiza os pesos e bias da camada escondida
        for h in range(self.hidden_neurons):
            for i in range(self.input_neurons):
                self.weights_ih[i][h] += learning_rate * hidden_deltas[h] * inputs[i]
            self.bias_h[h] += learning_rate * hidden_deltas[h]

# Exemplo de uso
nn = NeuralNetwork(2, 2, 1)
inputs = [0, 0]
targets = [0]

inputs = [0, 0]
targets = [0]
for epoch in range(100):
    nn.train(inputs, targets, 0.1)
print(nn.feedforward(inputs))

inputs = [0, 1]
targets = [0]
for epoch in range(100):
    nn.train(inputs, targets, 0.1)
print(nn.feedforward(inputs))

inputs = [1, 0]
targets = [0]
for epoch in range(100):
    nn.train(inputs, targets, 0.1)
print(nn.feedforward(inputs))

inputs = [1, 1]
targets = [1]
for epoch in range(100):
    nn.train(inputs, targets, 0.1)
print(nn.feedforward(inputs))

inputs = [0,1]

print(inputs)
output = nn.feedforward(inputs)
print(f'Output após treinamento: {output}')#/aaa