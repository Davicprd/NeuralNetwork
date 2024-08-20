import math
import random
input = 1
output_desire = 0
input_weidth = 0.01 * random.randint(0,100)
learning_rate = 0.01
count = 0
bias = 1
bias_weight = 0.01 * random.randint(0,100)
def activation(sum):
    if sum >= 0:
        return 1
    else:
        return 0
error = math.inf  
while error != 0:
    count += 1
    print("######## Iteração:", count)
    sum = input * input_weidth + (bias * bias_weight)
    output = activation(sum)
    error = output_desire -  output

    if error != 0:
        input_weidth = float(input_weidth + (learning_rate * input * error))
        bias_weight = bias_weight + learning_rate * bias * error

    print("Entrada:", input)
    print("Peso:",input_weidth)
    print("Peso Bias", bias_weight)
    print("Saida Desejada:", output_desire)
    print("Saida:", output)
    print("Erro:", error)