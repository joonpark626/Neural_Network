from copy import deepcopy
from operator import index
from re import X
import numpy as np
from math import *
import pandas as pd
import csv
from random import random

#Transfer neuron activation sigmoid function
def transfer_sigmoid(activation):
    return 1.0/(1.0 + exp(-activation))

def feedforward_comp(W_HL1_new, B_HL1_new, W_out_new, B_out_new, data_inputs):
    
    # print("---------------------------------------------------------------")
    # print("feedforward_comp: W_HL1_new = ", W_HL1_new)
    # print("feedforward_comp: B_HL1_new = ", B_HL1_new)
    # print("\n")
    # print("feedforward_comp: W_out_new = ", W_out_new)
    # print("feedforward_comp: B_out_new = ", B_out_new)
    # print("\n")
    # print("feedforward_comp: data_inputs = ", data_inputs)
    # print("\n")
    
    
    # Computation of hidden layer 1 neurons with weights and inputs
    HL1_comp = np.dot(data_inputs, W_HL1_new.T)
    #print("feedforward_comp: HL1_comp = ", HL1_comp)
    #print("\n")
    
    # Addition with the bias of hidden neurons
    SUM_HL1_comp = []
    for i in range(len(HL1_comp)):
        SUM_HL1_comp.append(np.array(HL1_comp[i]) + np.array(B_HL1_new))
    #print("feedforward_comp: SUM_HL1_comp = ", np.array(SUM_HL1_comp))
    #print("\n")
    
    # Applying the sigmoid activation function to activate all the elements (neuron outputs) for the data set
    Activated_SUM_HL1_comp = np.array([[transfer_sigmoid(x) for x in sample] for sample in deepcopy(SUM_HL1_comp)])
    print("feedforward_comp: Activated_SUM_HL1_comp = ", Activated_SUM_HL1_comp)
    print("\n")
    
    #print("feedforward_comp: W_out_new = ", W_out_new)
    #print("feedforward_comp: B_out_new = ", B_out_new)
    #print("\n")
           
    prev_calculated_outputs = np.dot(W_out_new, Activated_SUM_HL1_comp.T)
    #print("feedforward_comp: prev_calculated_outputs = ", prev_calculated_outputs)
    #print("feedforward_comp: prev_calculated_outputs.T TESTEST = ", prev_calculated_outputs.T)
    #print("\n")
    
    Transp_prev_calculated_outputs = np.transpose(prev_calculated_outputs)
    #print("feedforward_comp: Transp_prev_calculated_outputs TESTEST = ", Transp_prev_calculated_outputs)
    #print("\n")
    
    # Adding the list ("B_out_new") to each array element of "Transp_prev_calculated_outputs"
    # Element = one data sample
    final_calculated_outputs = np.add(Transp_prev_calculated_outputs, B_out_new)
    
    #print("feedforward_comp: final_calculated_outputs = ", final_calculated_outputs)
    #print("\n")
    
    return final_calculated_outputs, Activated_SUM_HL1_comp


# Comparing the calculated and measured outputs for conjugate gradient iteration
def error_checking(calculated_outputs, data_outputs):
    
    # print("-----------------------------------------------------------------")
    # print("error_checking: calculated_outputs = ", calculated_outputs)
    # print("error_checking: data_outputs = ", data_outputs)
    # print("\n")
    
    Y_max_min = max(calculated_outputs) - min(calculated_outputs)
    error_samples =  np.divide(np.subtract(calculated_outputs, data_outputs), Y_max_min)
    # print("error_checking: Y_max_min = ", Y_max_min)
    # print("error_checking: error_samples = ", error_samples)
    # print("\n")

    avg_error = np.sum(error_samples)/np.size(calculated_outputs)
    #print("error_checking: avg_error = ", avg_error)
    #print("\n")
    
    return error_samples, avg_error


def delta_W_HL1_computation(activated_hidden1_outputs, data_inputs, W_HL1_size):
    
    # About: To find the delta E of Weights in the hidden 1 layer. Since there are 12 samples to hanle
    #        array/matrix will be computed for one sample each for simplicity
    
    print("-----------------------------------------------------------------")
    print("delta_W_HL1_computation: activated_hidden1_outputs = ", activated_hidden1_outputs)
    print("delta_W_HL1_computation: data_inputs = ", data_inputs)
    print("delta_W_HL1_computation: W_HL1_size = ", W_HL1_size)
    print("\n")
    
    # need to make a matrix that is the same size matrix as W_HL1 and the matrix is the delta E function for each sample
    delta_w_HL1 = np.array([[pow(activated_hidden1_outputs[row], 2)*(1 - activated_hidden1_outputs[row])*data_inputs[col] for col in range(W_HL1_size[1])] for row in range(W_HL1_size[0])])
    print("delta_W_HL1_computation: delta_w_HL1 = ", delta_w_HL1)
    print("\n")
    
    return delta_w_HL1


def delta_W_function(SUM_de_w_out, SUM_de_B_out, SUM_de_w_HL1, SUM_de_B_HL1, Tr_errors):
    
    # About: Obtaining the derivative of the hidden layer 1 weights for each sample data
    # All the vectors, except the Tr_errors, are the Jacobian matrices
    print("-----------------------------------------------------------------")
    print("delta_W_function: SUM_de_w_out = ", SUM_de_w_out)
    print("delta_W_function: SUM_de_B_out = ", SUM_de_B_out)
    print("\n") 
    print("delta_W_function: SUM_de_w_HL1 = ", SUM_de_w_HL1)
    print("delta_W_function: SUM_de_B_HL1 = ", SUM_de_B_HL1)
    print("\n") 
    print("delta_W_function: Tr_errors = ", Tr_errors)
    print("\n")
    
    
    return delta_w_out, delta_B_out, delta_w_HL1, delta_B_HL1

def Jacobian_vector_for_HL1(W_HL1_new, B_HL1_new, W_out_new, B_out_new, activated_hidden1_outputs, data_inputs, calculated_outputs, data_outputs):
    
    # About: Obtaining the derivative of the hidden layer 1 weights for each sample data
    # print("-----------------------------------------------------------------")
    # print("Jacobian_vector_for_hidden1: W_HL1_new = ", W_HL1_new)
    # print("Jacobian_vector_for_hidden1: B_HL1_new = ", B_HL1_new)
    # print("\n")
    # print("Jacobian_vector_for_hidden1: W_out_new = ", W_out_new)
    # print("Jacobian_vector_for_hidden1: B_out_new = ", B_out_new)
    # print("\n")
    
    # print("Jacobian_vector_for_hidden1: data_inputs = ", data_inputs)
    # print("Jacobian_vector_for_hidden1: activated_hidden1_outputs = ", activated_hidden1_outputs)
    # print("\n")
    # print("Jacobian_vector_for_hidden1: calculated_outputs = ", calculated_outputs)
    # print("Jacobian_vector_for_hidden1: data_outputs = ", data_outputs)
    # print("\n")
    
    # Difference between calculated and measured outputs
    diff_calc_meas = np.add(np.array(data_outputs), calculated_outputs)
    #print("Jacobian_vector_for_hidden1: diff_calc_meas = ", diff_calc_meas)
    #print("\n")
    
    # ----------------------------------------------------------------------
    # Require to find the derivative of weights for all 12 samples
    de_B_out = np.ones((len(activated_hidden1_outputs), len(B_out_new)))
    de_w_out = np.array(deepcopy(activated_hidden1_outputs))
    
    # Determine the delta vector of hidden neuron biases
    de_B_HL1 = np.array([[(1-x)*pow(x,2) for x in sample] for sample in deepcopy(de_w_out)])
    
    # Find the matrix size of the hidden layer 1 weights
    W_HL1_size = np.shape(W_HL1_new)
    
    # Since a huge collection of arrays will be generated, for simplicity, created a function to allocate each data sample for finding delta dE/dw
    de_w_HL1 = np.array([None for i in range(len(activated_hidden1_outputs))])
    for i in range(len(activated_hidden1_outputs)):
        de_w_HL1[i] = delta_W_HL1_computation(activated_hidden1_outputs[i], data_inputs[i], W_HL1_size)
        
        
    # print("Jacobian_vector_for_hidden1: de_w_out = ", de_w_out)    
    # print("Jacobian_vector_for_hidden1: de_B_out = ", de_B_out)
    # print("\n")
    # print("Jacobian_vector_for_hidden1: de_B_HL1 = ", de_B_HL1)
    # print("\n")
    # print("Jacobian_vector_for_hidden1: de_w_HL1 = ", de_w_HL1)
    # print("\n")
    
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    SUM_de_w_out = de_w_out.sum(axis=0)
    SUM_de_B_out = np.array(np.sum(de_B_out))
    
    SUM_de_w_HL1 = de_w_HL1.sum(axis=0)
    SUM_de_B_HL1 = de_B_HL1.sum(axis=0)
    
    
    # print("Jacobian_vector_for_hidden1: SUM_de_w_out = ", SUM_de_w_out)
    # print("Jacobian_vector_for_hidden1: SUM_de_B_out = ", SUM_de_B_out)
    # print("\n") 
    # print("Jacobian_vector_for_hidden1: SUM_de_w_HL1 = ", SUM_de_w_HL1)
    # print("Jacobian_vector_for_hidden1: SUM_de_B_HL1 = ", SUM_de_B_HL1)
    # print("\n") 
        
    return SUM_de_w_out, SUM_de_B_out, SUM_de_w_HL1, SUM_de_B_HL1


def Levenberg_Marquardt_HL1(user_Validation_error, W_HL1, B_HL1, W_out, B_out, Tr_data_inputs, Tr_data_outputs, V_data_inputs, V_data_outputs, Te_data_inputs, Te_data_outputs, mui, max_epoch):
    
    Training_error_collection = [] 
    num_epoch_iterated = 0
    
    W_HL1_new = W_HL1
    W_out_new = W_out
    B_HL1_new = B_HL1
    B_out_new = B_out
    
    # print("Levenberg_Marquardt : W_HL1_new = ", W_HL1_new)
    # print("Levenberg_Marquardt : W_out_new = ", W_out_new)
    # print("Levenberg_Marquardt : B_HL1_new = ", B_HL1_new)
    # print("Levenberg_Marquardt : B_out_new = ", B_out_new)
    # print("\n")
    
    # print("Levenberg_Marquardt : data_Tr_inputs = ", Tr_data_inputs)
    # print("Levenberg_Marquardt : data_Tr_outputs = ", Tr_data_outputs)
    # print("\n")
    # print("Levenberg_Marquardt : data_V_inputs = ", V_data_inputs)
    # print("Levenberg_Marquardt : data_V_outputs = ", V_data_outputs)
    # print("\n")
    # print("Levenberg_Marquardt : data_TE_inputs = ", Te_data_inputs)
    # print("Levenberg_Marquardt : data_Te_outputs = ", Te_data_outputs)
    # print("\n")

    
    for epoch in range(max_epoch):
        
        Tr_calculated_outputs, activated_hidden1_outputs = feedforward_comp(W_HL1_new, B_HL1_new, W_out_new, B_out_new, Tr_data_inputs)
        Tr_errors, Tr_avg_error = error_checking(Tr_calculated_outputs, Tr_data_outputs)

        # print("Levenberg_Marquardt : Tr_calculated_outputs = ", Tr_calculated_outputs)
        # print("Levenberg_Marquardt : activated_hidden1_outputs = ", activated_hidden1_outputs)
        # print("Levenberg_Marquardt : Tr_errors = ", Tr_errors)
        # print("Levenberg_Marquardt : Tr_avg_error = ", Tr_avg_error)
        # print("\n")
        
        V_calculated_outputs, dummy = feedforward_comp(W_HL1_new, B_HL1_new, W_out_new, B_out_new, V_data_inputs)
        V_errors, V_avg_error = error_checking(V_calculated_outputs, V_data_outputs)
        
        # print("Levenberg_Marquardt : V_calculated_outputs = ", V_calculated_outputs)
        # print("Levenberg_Marquardt : V_errors = ", V_errors)
        # print("Levenberg_Marquardt : V_avg_error = ", V_avg_error)
        # print("\n")
        
        if(Tr_avg_error < user_Validation_error or V_avg_error == Tr_avg_error):
            break
        
        SUM_de_w_out, SUM_de_B_out, SUM_de_w_HL1, SUM_de_B_HL1 = Jacobian_vector_for_HL1(W_HL1_new, B_HL1_new, W_out_new, B_out_new, activated_hidden1_outputs, Tr_data_inputs, Tr_calculated_outputs, Tr_data_outputs)
        
        delta_w_out, delta_B_out, delta_w_HL1, delta_B_HL1 = delta_W_function(SUM_de_w_out, SUM_de_B_out, SUM_de_w_HL1, SUM_de_B_HL1, Tr_errors)
        
        
        num_epoch_iterated = num_epoch_iterated + 1
        #print("conjugate_gradient: num_epoch_iterated (epoch >= ",epoch,") = ", num_epoch_iterated)    
    calculated_outputs_Te = forward_propagate(W_new_out[0], W_new_HLN1[0], data_Te_input_Vgs, data_Te_input_Vds)
    FF_error_Te = error_checking(calculated_outputs_Te, data_Te_output_Id)

    return W_HL1_new, B_HL1_new, W_out_new, B_out_new, num_epoch_iter, Training_error_collection, FF_error_Te

#-------------------------------------------------------------------------------------
# Initial Neural Network Attributes setup

# Printing the values of each sample but removing the first element of list which is the name of different values
with open('MOS_Tr.csv', newline='') as f:
    reader = csv.reader(f)
    data_Tr = list(reader)
data_Tr.pop(0)

with open('MOS_V.csv', newline='') as f:
    reader = csv.reader(f)
    data_V = list(reader)
data_V.pop(0)

with open('MOS_Te.csv', newline='') as f:
    reader = csv.reader(f)
    data_Te = list(reader)
data_Te.pop(0)

Ali_num_inputs = 2
Ali_num_outputs = 1
Ali_num_HL = 1
Ali_HL1_neuron = 3
Ali_HL2_neuron = 3
Ali_HL3_neuron = 3
Ali_Mui = 0.1

# Converting the strings of string elememts of each sample into float, what float? because there are negative values
for sample in data_Tr:
    for i in range(len(sample)):
        sample[i] = float(sample[i])
        
for sample2 in data_V:
    for i in range(len(sample)):
        sample2[i] = float(sample2[i])

for sample3 in data_Te:
    for i in range(len(sample3)):
        sample3[i] = float(sample3[i])

print("data_Tr = ", data_Tr)
print("data_V = ", data_V)
print("data_Te = ", data_Te)
print("\n")

# ---------------------------------------------------
# Data Generation and scaling of Training set
temp_data_Tr_inputs = []
temp_data_Tr_outputs = []
temp_data_V_inputs = []
temp_data_V_outputs = []
temp_data_Te_inputs = []
temp_data_Te_outputs = []

data_Tr_inputs = []
data_Tr_outputs = []
data_V_inputs = []
data_V_outputs = []
data_Te_inputs = []
data_Te_outputs = []

for sample in data_Tr:
    for j in range(len(sample)):
        if (j < Ali_num_inputs):
            temp_data_Tr_inputs.extend([sample[j]])

        else:
            temp_data_Tr_outputs.extend([sample[j]])
    data_Tr_inputs.append(deepcopy(temp_data_Tr_inputs))
    data_Tr_outputs.append(deepcopy(temp_data_Tr_outputs))
    temp_data_Tr_inputs.clear()
    temp_data_Tr_outputs.clear()

for sample2 in data_V:
    for j in range(len(sample2)):
        if (j < Ali_num_inputs):
            temp_data_V_inputs.extend([sample2[j]])
        else:
            temp_data_V_outputs.extend([sample2[j]])
    data_V_inputs.append(deepcopy(temp_data_V_inputs))
    data_V_outputs.append(deepcopy(temp_data_V_outputs))
    temp_data_V_inputs.clear()
    temp_data_V_outputs.clear()

for sample3 in data_Te:
    for j in range(len(sample3)):
        if (j < Ali_num_inputs):
            temp_data_Te_inputs.extend([sample3[j]])
        else:
            temp_data_Te_outputs.extend([sample3[j]])
    data_Te_inputs.append(deepcopy(temp_data_Te_inputs))
    data_Te_outputs.append(deepcopy(temp_data_Te_outputs))
    temp_data_Te_inputs.clear()
    temp_data_Te_outputs.clear()

data_Tr_inputs = np.array(data_Tr_inputs)
data_Tr_outputs = np.array(data_Tr_outputs)
data_V_inputs = np.array(data_V_inputs)
data_V_outputs = np.array(data_V_outputs)
data_Te_inputs = np.array(data_Te_inputs)
data_Te_outputs = np.array(data_Te_outputs)

print("data_Tr_inputs = ", data_Tr_inputs)
print("data_Tr_outputs = ", data_Tr_outputs)
print("\n")
print("data_V_inputs = ", data_V_inputs)
print("data_V_outputs = ", data_V_outputs)
print("\n")
print("data_Te_inputs = ", data_Te_inputs)
print("data_Te_outputs = ", data_Te_outputs)
print("\n")

# Initialize the weight values when a number was typed by an user in GUI

max_epoch = 3;
user_Validation_error = 0.1;
mui = Ali_Mui

if (Ali_num_HL == 1):
    W_HL1 = np.random.rand(Ali_HL1_neuron, Ali_num_inputs)
    B_HL1 = np.random.rand(Ali_HL1_neuron)
    W_out = np.random.rand(Ali_num_outputs, len(W_HL1))
    B_out = np.random.rand(Ali_num_outputs)
    [W_HL1_new, B_HL1_new, W_out_new, B_out_new, num_epoch_iter, Training_error_collection, FF_error_Te] = Levenberg_Marquardt_HL1(user_Validation_error, W_HL1, B_HL1, W_out, B_out, data_Tr_inputs, data_Tr_outputs, data_V_inputs, data_V_outputs, data_Te_inputs, data_Te_outputs, mui, max_epoch)
elif (Ali_num_HL == 2):
    W_HL1 = np.random.rand(len(data_Te), Ali_num_inputs)
    W_HL2 = np.random.rand(len(data_Te), Ali_num_inputs)
print("W_HL1 = ", W_HL1)
print("B_HL1 = ", B_HL1)
print("W_out = ", W_out)
print("B_out = ", B_out)
print("\n")

#-----------------------NOTE----------------------------
#The structure of the ANN with one hidden layer
#                    Id
#        # of Hidden Neurons here
#                Vgs     Vds


#note ali's input = 3
# data_Tr_input_V1 = []
# data_Tr_input_V2 = []
# data_Tr_input_V3 = []
# data_Tr_input_V4 = []
# data_Tr_input_V5 = []
# data_Tr_input_V6 = []
# data_Tr_input_V7 = []
# data_Tr_input_V8 = []
# data_Tr_input_V9 = []
# data_Tr_input_V10 = []
