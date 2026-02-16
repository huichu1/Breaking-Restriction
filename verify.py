import numpy as np
DIM_LIST = [20,15,10,5]
IDIM = 5
DIM1 = 4
DIM2 = 3
DIM3 = 2

A1 = np.random.randn(DIM_LIST[1],DIM_LIST[0])
B1 = np.random.randn(DIM_LIST[1])
S1 = np.random.uniform(0.1,2,(DIM_LIST[1]))

A2 = np.random.randn(DIM_LIST[2],DIM_LIST[1])
B2 = np.random.randn(DIM_LIST[2])
S2 = np.random.uniform(0.1,2,(DIM_LIST[2]))

A3 = np.random.randn(DIM_LIST[3],DIM_LIST[2])
B3 = np.random.randn(DIM_LIST[3])
print("1 layer is")
print(A1)
print(B1)
print(S1)

print("2 layer is")
print(A2)
print(B2)
print(S2)

print("3 layer is")
print(A3)
print(B3)

def forward_original(x):
    y1 = A1@x + B1
    for index in range(len(y1)):
        if y1[index] < 0:
            y1[index] = y1[index] * S1[index]
    y2 = A2@y1 + B2
    for index in range(len(y2)):
        if y2[index] < 0:
            y2[index] = y2[index] * S2[index]
    y3 = A3@y2 + B3
    return y3

def transfer_weight(A1,B1,S1,A2,B2,S2,A3,B3):
    new_A1 = A1.copy()
    new_B1 = B1.copy()
    new_S1 = S1.copy()
    new_A2 = A2.copy()
    new_B2 = B2.copy()
    new_S2 = S2.copy()
    new_A3 = A3.copy()
    new_B3 = B3.copy()

    transfer_1 = np.random.uniform(-10,10,(DIM_LIST[1]))
    transfer_2 = np.random.uniform(-10,10,(DIM_LIST[2]))

    for i in range(len(transfer_1)):
        new_A1[i] = new_A1[i] * transfer_1[i]
        new_B1[i] = new_B1[i] * transfer_1[i]
        if transfer_1[i] < 0:
            new_S1[i] = 1/S1[i]
        
        if transfer_1[i] >= 0:
            new_A2[:,i] = new_A2[:,i] * (1/transfer_1[i])
        else:
            new_A2[:,i] = new_A2[:,i] * (S1[i]/transfer_1[i])
        
    for i in range(len(transfer_2)):
        new_A2[i] = new_A2[i] * transfer_2[i]
        new_B2[i] = new_B2[i] * transfer_2[i]
        if transfer_2[i] < 0:
            new_S2[i] = 1/S2[i]
        
        if transfer_2[i] >= 0:
            new_A3[:,i] = new_A3[:,i] * (1/transfer_2[i])
        else:
            new_A3[:,i] = new_A3[:,i] * (S2[i]/transfer_2[i])
    return new_A1,new_B1,new_S1,new_A2,new_B2,new_S2,new_A3,new_B3
new_A1,new_B1,new_S1,new_A2,new_B2,new_S2,new_A3,new_B3 = transfer_weight(A1,B1,S1,A2,B2,S2,A3,B3)
print("transfer1 layer is")
print(new_A1)
print(new_B1)
print(new_S1)

print("transfer2 layer is")
print(new_A2)
print(new_B2)
print(new_S2)

print("transfer3 layer is")
print(new_A3)
print(new_B3)

def forward_haha(x):
    y1 = new_A1@x + new_B1
    for index in range(len(y1)):
        if y1[index] < 0:
            y1[index] = y1[index] * new_S1[index]
    y2 = new_A2@y1 + new_B2
    for index in range(len(y2)):
        if y2[index] < 0:
            y2[index] = y2[index] * new_S2[index]
    y3 = new_A3@y2 + new_B3
    return y3

input_x = np.random.randn(DIM_LIST[0])

result_1 = forward_original(input_x)
result_2 = forward_haha(input_x)
print("check result!")
print(result_1)
print(result_2)