import random
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, LayerId, ID, ActivateFunc = 0):
        self.Layer = LayerId
        self.ID = ID

        self.Input = 0
        self.Output = 0
        self.Loss = 0

        # Set Activation Function
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

    # This function create connection between layers
    def defineConnections(self, ConnectedLayer):
        self.Connections = []

        # Set Connection between every neuron between two layers, and set random Weight
        for TN in ConnectedLayer.Neurons:
            for SN in self.Neurons:
                self.Connections.append([SN,TN, random.random() - .5])

    # This function display every connection every neuron in this layer
    def showConnections(self):
        for _ in range(len(self.Connections)):
            print(f"{_} Source Neuron {self.Connections[_][0].ID} from Layer {self.Connections[_][0].Layer} "
                  f"is connected with Target Neuron {self.Connections[_][1].ID} from Layer {self.Connections[_][1].Layer} "
                  f"with weight {self.Connections[_][2]}")

    # This function sum and weights every input for Neuron with ID = TargetNeuronID  in this Layer
    def sumValues(self, TargetNeuronID):
        Sum = 0
        supp = len(self.Neurons)*TargetNeuronID
        for _ in range(len(self.Neurons)):
            Sum = Sum + self.Connections[_ + supp][0].Output * self.Connections[_ + supp][2]
        self.Connections[supp][1].Input = Sum


# Main Class
class NeuralNetwork:

    def __init__(self, Layers, DropOut = 0, LearningRate=0.001):
        self.Layers = Layers
        self.DropOut = DropOut
        self.LearningRate = LearningRate

        #W każdej warstwie utwórz połączenia neuronowe każdy z każdym
        for L in range(len(self.Layers)-1):
            self.Layers[L].defineConnections(self.Layers[L+1])

    def runNetwork(self, Input, dropOut = False):

        # Przypisz wszystkie inputy do warstwy wejściowej jako output
        for N in range(len(self.Layers[0].Neurons)):
            if len(Input) > N:
                self.Layers[0].Neurons[N].Output = Input[N]
            else:
                self.Layers[0].Neurons[N].Output = 0


        # Zsumuj i zważ, a potem wykonaj odpowiednią funkcje aktywacji
        for L in range(len(self.Layers)-1):
            for N in range(len(self.Layers[L+1].Neurons)):
                self.Layers[L].sumValues(N)
                if (random.random() < self.DropOut) & dropOut:
                    self.Layers[L + 1].Neurons[N].Input = 0
                self.Layers[L+1].Neurons[N].ActivateFunc()


        return self.Layers[len(self.Layers)-1].Neurons[0].Output

    # Learning by BackPropagation
    def BackPropagation(self, Input, Target):
        Actual = self.runNetwork(Input, True)
        # Stop this run if Loss is equal to 0
        if Input == Target:
            return
        else:
            # Calculate Loss to Output Neuron
            Loss = Target - Actual
            self.Layers[len(self.Layers)-1].Neurons[0].Loss = Loss

            # Calculate Loss for Every Neuron in Every layer based on first Loss
            for L in range(len(self.Layers)-1):
                LN = len(self.Layers)-L-2
                for N in range(len(self.Layers[LN].Connections)):
                    self.Layers[LN].Connections[N][0].Loss = self.Layers[LN].Connections[N][0].Loss + \
                                            self.Layers[LN].Connections[N][1].Loss * self.Layers[LN].Connections[N][2]

            # Calculate new weight for every single Neuron
            for L in range(len(self.Layers)-2):
                for N in range(len(self.Layers[L].Connections)):
                    self.Layers[L].Connections[N][2] += self.Layers[L].Connections[N][1].Loss * self.LearningRate * \
                                self.Layers[L].Connections[N][0].Output
                    # Important! Clear Loss
                    self.Layers[L].Connections[N][1].Loss = 0

L = [Layer(2,0),Layer(15,1), Layer(15,2), Layer(1,3)]

P = NeuralNetwork(L, LearningRate=0.1)
P.DropOut = 0.00

# Test for Neural network. Draw 1000 time 2 number in range 0 - 5
Test = []
for _ in range(1100):
    Test.append([random.random(), random.random()])

# Check last 10 additions on untrained nn
for _ in range(10):
    Test[1000 + _][0] = random.random()*1000
    Test[1000 + _][1] = random.random()*1000
    print(f"{Test[1000+_][0]+Test[1000+_][1]} == {P.runNetwork(Test[1000+_])}")


Loss = []
x = 2
# Leeaaaarrrnnnnnniiiiiinnnngggg
for _ in range(30000):
    r = random.randint(0,100)
    P.BackPropagation(Test[r],Test[r][0]+Test[r][1])

    if _ != 0:
        if _%100 == 0:
            x+=1
    if _%(1*x) == 0:
        L = 1
        for __ in range(50):
            r = random.randint(101,500)
            L = L + (Test[r][0]+Test[r][1] - P.runNetwork(Test[r]))**2
        L = L/50
        Loss.append(L)
plt.ylim([0,1])
plt.plot(range(len(Loss)),Loss)

# Check last 10 additions on trained nn
print("")
for _ in range(10):
    print(f"{Test[1000+_][0]+Test[1000+_][1]} == {P.runNetwork(Test[1000+_])}")

plt.show()
# Nice Result:
# 6.784353321416075 == -0.13083023941535152
# 6.885501468226714 == -0.20796350121445006
# 5.596649153547185 == -0.1681687268052153
# 8.031263445748227 == -0.2548802423653442
# 4.946491356204765 == -0.13701135131164766
# 5.085719976501892 == -0.13952735895112495
# 7.321720455734509 == -0.16562475963050602
# 6.514129753039228 == -0.0988311790763748
# 5.949313651011419 == -0.27786715843065063
# 5.035096211012926 == -0.13887249975719806

# 6.784353321416075 == 6.784353321416088
# 6.885501468226714 == 6.8855014682267095
# 5.596649153547185 == 5.596649153547182
# 8.031263445748227 == 8.03126344574822
# 4.946491356204765 == 4.946491356204766
# 5.085719976501892 == 5.085719976501891
# 7.321720455734509 == 7.321720455734516
# 6.514129753039228 == 6.514129753039245
# 5.949313651011419 == 5.949313651011396
# 5.035096211012926 == 5.035096211012925