import numpy as np 
import matplotlib.pyplot as plt 

def sigmoid(x):
    return 1 / (1+ np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

x = np.random.randn(1000, 100) # 1000개의 데이터 
node_num = 100        # 은닉층의 뉴런 수 
hidden_layer_size = 5 # 은닉층이 5개 
activations = {}      # 활성화 결과(값)을 저장 

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    #w = np.random.randn(node_num, node_num) * 1
    #w = np.random.randn(node_num, node_num) * 0.01
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)

    a = np.dot(x, w)
    #z = sigmoid(a)
    z = tanh(x)
    activations[i] = z 

# 히스토그램 그리기 
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30, range = (0,1))

plt.show()
    