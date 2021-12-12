import random
import numpy as np
import matplotlib.pyplot as plt

INPUT_DIM = 3
OUT_DIM = 3
H_DIM = 9
ALPHA = 0.1
#BATCH_SIZE = 3

def relu(t):
    return np.maximum(t, 0)

def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)

def sparse_cross_entropy(z, y):
    return -np.log(z[0, y])

def relu_deriv(t):
    return (t >= 0).astype(float)
# def softmax_batch(t):
#     out = np.exp(t)
#     return out / np.sum(out, axis=1, keepdims=True)

# def sparse_cross_entropy_batch(z, y):
#     return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))

# def to_full(y, num_classes):
#     y_full = np.zeros((1, num_classes))
#     y_full[0, y] = 1
#     return y_full

# def to_full_batch(y, num_classes):
#     y_full = np.zeros((len(y), num_classes))
#     for j, yj in enumerate(y):
#         y_full[j, yj] = 1
#     return y_full

class NeuralNetwork:
    def __init__(self):

        self.W1 = np.random.rand(INPUT_DIM, H_DIM)
        self.b1 = np.random.rand(1, H_DIM)
        self.W2 = np.random.rand(H_DIM, OUT_DIM)
        self.b2 = np.random.rand(1, OUT_DIM)

    def forward(self,x):
        self.x =x
        t1 = x @ self.W1 + self.b1
        h1 = relu(t1)
        t2 = h1 @ self.W2 + self.b2
        z = softmax(t2)
        return z

    def backprop(self,z,x,y):
        y_full = np.zeros((1, OUT_DIM))
        y_full[0, y] = 1
        df = z - y_full
        t1=x @ self.W1 + self.b1
        self.W1-=ALPHA*x.T@(df@self.W2.T*relu_deriv(t1))
        self.b1-=ALPHA*df@self.W2.T*relu_deriv(t1)
        self.W2-=ALPHA*relu(t1).T@df
        self.b2-=ALPHA*df
    def train(self,x,y):

        self.backprop(self.forward(x),x,y)

dataset=[]
X = np.array([[[1, 1, 0]], [[1, 0, 1]], [[0, 1, 1]], [[0, 1, 0]],[[0,1,1]],[[0,0,1]],[[0,1,0]],[[1,1,1]],[[0,0,0]]])
y = np.array((0, 0, 1, 2,1, 2, 2, 0,1))
for i in range(len(y)):
    dataset.append((X[i],y[i]))

loss_arr = []
loss=1
NN=NeuralNetwork()
# for ep in range(NUM_EPOCHS):
while (loss/len(dataset)>0.01):
    loss=0
    random.shuffle(dataset)
    for i in range(len(dataset)):
        x,y = dataset[i]
        NN.train(x,y)
        loss+=sparse_cross_entropy(NN.forward(x),y)
    loss_arr.append(loss/len(dataset))
plt.plot(loss_arr)
plt.show()

def acc():
    z=0
    print("Table check")
    for i in range(len(dataset)):
        x, y = dataset[i]
        print(x)
        y1=np.argmax(NN.forward(x))
        print("Class",y1+1)
        if(y1==y):
            z+=1
    print("Accuracy:",z/len(dataset))
dataset.append(([[1,0,0]], 0))
acc()
a=[]
print("Get example")
for i in range(3):
    a.append(int(input()))
print("Class:",np.argmax(NN.forward(a))+1)


