import random

class Neuron:
    def __init__(self, LayerId, ID, ActivateFunc = 0):
        self.Layer = LayerId
        self.ID = ID

        self.Input = 0
        self.Output = 0
        self.Loss = 0

        #Ustaw funkcje aktywacji
        self.AFunc = ActivateFunc
        if self.AFunc == 0:
            self.ActivateFunc = self.linear
        elif self.AFunc == 1:
            self.ActivateFunc = self.ReLU

    def linear(self):
        self.Output = self.Input

    def ReLU(self):
        self.Output = max(0, self.Input)


class Layer:
    def __init__(self, Neurons, LayerId):
        self.LayerId = LayerId
        self.Neurons = [Neuron(LayerId, _) for _ in range(Neurons)]

    #Ta funkcja tworzy połączenia między warstwami
    def defineConnections(self, ConnectedLayer):
        self.Connections = []

        #Każdy neuron źródłowy
        for TN in ConnectedLayer.Neurons:
            for SN in self.Neurons:
                self.Connections.append([SN,TN, random.random() - .5])

    #Ta funkcja wyświetla wszystkie połączenia wszystkich neuronów wewnątrz tej warstwy z kolejnymi
    def showConnections(self):
        for _ in range(len(self.Connections)):
            print(f"{_} Source Neuron {self.Connections[_][0].ID} from Layer {self.Connections[_][0].Layer} "
                  f"is connected with Target Neuron {self.Connections[_][1].ID} from Layer {self.Connections[_][1].Layer} "
                  f"with weight {self.Connections[_][2]}")

    #Ta funkcja wykonuje sumowanie i ważenie kolejnych wejść do danego Neurona
    def sumValues(self, TargetNeuronID):
        Sum = 0
        supp = len(self.Neurons)*TargetNeuronID
        for _ in range(len(self.Neurons)):
            Sum = Sum + self.Connections[_ + supp][0].Output * self.Connections[_ + supp][2]
        self.Connections[supp][1].Input = Sum


class NeuralNetwork:

    def __init__(self, Layers, DropOut = 0, LearningRate=0.01):
        self.Layers = Layers
        self.DropOut = DropOut
        self.LearningRate = LearningRate

        #W każdej warstwie utwórz połączenia neuronowe każdy z każdym
        for L in range(len(self.Layers)-1):
            self.Layers[L].defineConnections(self.Layers[L+1])

    def runNetwork(self, Input):

        #Przypisz wszystkie inputy do warstwy wejściowej jako output
        for N in range(len(self.Layers[0].Neurons)):
            if len(Input) > N:
                self.Layers[0].Neurons[N].Output = Input[N]
            else:
                self.Layers[0].Neurons[N].Output = 0


        #Zsumuj i zważ, a potem wykonaj odpowiednią funkcje aktywacji
        for L in range(len(self.Layers)-1):
            for N in range(len(self.Layers[L+1].Neurons)):
                self.Layers[L].sumValues(N)
                self.Layers[L+1].Neurons[N].ActivateFunc()


        return self.Layers[len(self.Layers)-1].Neurons[0].Output

    def BackPropagation(self, Input, Target):
        Actual = self.runNetwork(Input)
        if Input == Target:
            return
        else:
            Loss = Target - Actual
            self.Layers[len(self.Layers)-1].Neurons[0].Loss = Loss

            for L in range(len(self.Layers)-1):
                LN = len(self.Layers)-L-2
                for N in range(len(self.Layers[LN].Connections)):
                    self.Layers[LN].Connections[N][0].Loss = self.Layers[LN].Connections[N][0].Loss + \
                                            self.Layers[LN].Connections[N][1].Loss * self.Layers[LN].Connections[N][2]

            for L in range(len(self.Layers)-2):
                for N in range(len(self.Layers[L].Connections)):
                    self.Layers[L].Connections[N][2] += self.Layers[L].Connections[N][1].Loss * self.LearningRate * \
                                self.Layers[L].Connections[N][0].Output
                    self.Layers[L].Connections[N][1].Loss = 0

L = [Layer(2,0),Layer(5,1), Layer(5,2), Layer(1,3)]

P = NeuralNetwork(L)

#print(P.runNetwork([3,4]))

Test = []
for _ in range(1000):
    Test.append([random.random()*5, random.random()*5])


for _ in range(10):
    print(f"{Test[989+_][0]+Test[989+_][1]} == {P.runNetwork(Test[989+_])}")

for _ in range(25000):
    r = random.randint(0,100)
    P.BackPropagation(Test[r],Test[r][0]+Test[r][1])

print("")
for _ in range(10):
    print(f"{Test[989+_][0]+Test[989+_][1]} == {P.runNetwork(Test[989+_])}")