# Algoritmo baseado no livro Deep Learning Book, encontrado em http://www.deeplearningbook.com.br/

import numpy as np
import random
import csv as csv
import os
from sklearn.model_selection import KFold

# Classe de Rede Neural

class Network(object):

    # Função de inicialização da classe
    def __init__(self,sizes):
        # A lista sizes funciona com base no tamanho da rede. Uma lista [1,1,1] cria três camadas com um neurônio em cada camada.
        self.num_layers = len(sizes)
        self.sizes = sizes
        #Gerando aleatoriamente os biases iniciais, menos para a camada de entrada
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # Função para retornar a saída da rede, se a variável 'a' conter dados de entrada
    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a
    
    # Função para cálculo do stochastic gradient descent usando mini-batch
    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data = None):
        # training_data -> tuples com valores de x e y
        # epochs-> número de épocas 
        # mini_batch_size -> tamanho dos volumes de dados que serão processados
        # eta -> taxa de aprendizagem
        # test_data -> dados para realização de teste de acurácia, não obrigatório
        
        training_data = list(training_data)
        n = len(training_data)

        # Verificando se temos dados de teste 
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            
            if test_data:
                acertos = self.classify(test_data)
                print("Época {} / {} acertos em {} testes  / Acurácia {} %".format(j,acertos,n_test, int(acertos / n_test *100)))
            else:
                print("Época {} finalizada ".format(j))

    
    # Função para atualização e aplicação do SGD em pequenos lotes e realizando o backpropagation
    def update_mini_batch(self,mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Retorna uma tupla `(nabla_b, nabla_w)` representando o
            gradiente para a função de custo C_x. `nabla_b` e
            `nabla_w` são listas de camadas de matrizes numpy, semelhantes
            a `self.biases` e `self.weights`."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Feedforward
        activation = x

        # Lista para armazenar todas as ativações, camada por camada
        activations = [x] 

        # Lista para armazenar todos os vetores z, camada por camada
        zs = [] 

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # Aqui, l = 1 significa a última camada de neurônios, l = 2 é a segunda e assim por diante. 
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def classify(self, test_data):
        """Retorna o número de entradas de teste para as quais a rede neural 
         produz o resultado correto. Note que a saída da rede neural
         é considerada o índice de qualquer que seja
         neurônio na camada final que tenha a maior ativação."""

        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Retorna o vetor das derivadas parciais."""
        return (output_activations-y)


# Função de Ativação Sigmóide
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# Função para retornar as derivadas da função Sigmóide
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


#Carregando o dataset 
def load_dataset():
    file = "OCRdata.csv"
    with open(file) as dataset:
        data = np.array(list(csv.reader(dataset)))

        #Removendo linhas duplicadas depois do carregamento do arquivo
        data = np.unique(data,axis=0)

        #Separando os dados
        x_data = np.zeros((len(data)-1, len(data[0])-1))
        y_data = np.empty(len(data)-1)

        # Alimentando os arrays de dados
        for _ in range(0,len(data)-1):
            x_data = data[:,1:len(data)-1]
            y_data = data[:,0]

    #Retornando o dataset configurado e transformado em valores float
    return x_data.astype(int),y_data.astype(int) 
    

x_data,y_data = load_dataset() # Populando variáveis com os dados X e Y

#Criando a rede neural, 64 neurônios de entrada, 40 na camada oculta, 10 neurônios na camada de saída
net = Network([64,40,10])

#Inicializando o K-Fold
kf = KFold(n_splits=10,shuffle=True,random_state=1)

#Função para vetorizar os dados de classes
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[int(j)] = 1.0
    return e

# Realizando os K-Fold tanto para testes quanto para treinamento
for train_indices, test_indices in kf.split(x_data,y_data):

    # Criando os dados de treinamento 
    training_inputs = [np.reshape(x, (64, 1)) for x in x_data[train_indices]]
    training_results = [vectorized_result(y) for y in y_data[train_indices]]
    training_data = zip(training_inputs,  training_results)

    # Criando os dados para testes
    test_inputs = [np.reshape(x, (64, 1)) for x in x_data[test_indices]]   
    test_data = zip(test_inputs, y_data[test_indices])

    #Treinamento a rede
    net.SGD(training_data, 1000, 10, 3,test_data=test_data)

