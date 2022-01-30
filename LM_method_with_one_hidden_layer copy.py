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
    print("feedforward_comp: HL1_comp = ", HL1_comp)
    print("\n")
    
    # Addition with the bias of hidden neurons
    SUM_HL1_comp = []
    for i in range(len(HL1_comp)):
        SUM_HL1_comp.append(np.array(HL1_comp[i]) + np.array(B_HL1_new))
    print("feedforward_comp: SUM_HL1_comp = ", np.array(SUM_HL1_comp))
    print("\n")
    
    # Applying the sigmoid activation function to activate all the elements (neuron outputs) for the data set
    Activated_SUM_HL1_comp = np.array([[transfer_sigmoid(x) for x in sample] for sample in deepcopy(SUM_HL1_comp)])
    print("feedforward_comp: Activated_SUM_HL1_comp = ", Activated_SUM_HL1_comp)
    print("\n")
    
    print("feedforward_comp: W_out_new = ", W_out_new)
    print("feedforward_comp: B_out_new = ", B_out_new)
    print("\n")
           
    prev_calculated_outputs = np.dot(W_out_new, Activated_SUM_HL1_comp.T)
    print("feedforward_comp: prev_calculated_outputs = ", prev_calculated_outputs)
    #print("feedforward_comp: prev_calculated_outputs.T TESTEST = ", prev_calculated_outputs.T)
    print("\n")
    
    Transp_prev_calculated_outputs = np.transpose(prev_calculated_outputs)
    print("feedforward_comp: Transp_prev_calculated_outputs TESTEST = ", Transp_prev_calculated_outputs)
    print("\n")
    
    # Adding the list ("B_out_new") to each array element of "Transp_prev_calculated_outputs"
    # Element = one data sample
    final_calculated_outputs = np.add(Transp_prev_calculated_outputs, B_out_new)
    
    print("feedforward_comp: final_calculated_outputs = ", final_calculated_outputs)
    print("\n")
    
    return final_calculated_outputs


def activated_neuron_function(W_HL1_new, B_HL1_new, W_out_new, B_out_new, data_inputs):
    
    # About: Code segment from feedforward_comp will be reused for obtaining the hidden neurons' activated outputs
    
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
    activated_hidden_outputs = np.array([[transfer_sigmoid(x) for x in sample] for sample in deepcopy(SUM_HL1_comp)])
    
    return activated_hidden_outputs

# Comparing the calculated and measured outputs for conjugate gradient iteration
def error_checking(calculated_outputs, data_outputs):
    
    # print("-----------------------------------------------------------------")
    # print("error_checking: calculated_outputs = ", calculated_outputs)
    # print("error_checking: data_outputs = ", data_outputs)
    # print("\n")
    
    
    for i in range(len(calculated_outputs)):
        error_samples = (calculated_outputs[i] - data_outputs[i])/(np.max(calculated_outputs) - np.min(calculated_outputs))
    #print("error_checking: error_samples = ", error_samples)
    #print("\n")
    
    sum_error = sum(error_samples)
    avg_error = sum_error/len(calculated_outputs[0])
    # print("error_checking: sum_error = ", sum_error)
    # print("error_checking: avg_error = ", avg_error)
    # print("\n")
    
    return error_samples, avg_error

def Jacobian_vector_for_HL1(W_HL1_new, B_HL1_new, W_out_new, B_out_new, activated_hidden1_outputs, data_inputs, calculated_outputs, data_outputs):
    
    # print("-----------------------------------------------------------------")
    # print("Jacobian_vector_for_hidden1: W_HL1_new = ", W_HL1_new)
    # print("Jacobian_vector_for_hidden1: B_HL1_new = ", B_HL1_new)
    # print("\n")
    # print("Jacobian_vector_for_hidden1: W_out_new = ", W_out_new)
    # print("Jacobian_vector_for_hidden1: B_out_new = ", B_out_new)
    # print("\n")
    # print("Jacobian_vector_for_hidden1: activated_hidden1_outputs = ", activated_hidden1_outputs)
    # print("Jacobian_vector_for_hidden1: calculated_outputs = ", calculated_outputs)
    # print("Jacobian_vector_for_hidden1: data_outputs = ", data_outputs)
    # print("\n")
    
    # Difference between calculated and measured outputs
    diff_calc_meas = np.add(np.array(data_outputs), calculated_outputs)
    #print("Jacobian_vector_for_hidden1: diff_calc_meas = ", diff_calc_meas)
    #print("\n")
    
    # Require to find the derivative of weights for all 12 samples
    delta_B_out = np.ones((len(activated_hidden1_outputs), len(B_out_new)))
    delta_w_out = np.array(deepcopy(activated_hidden1_outputs))
    #print("Jacobian_vector_for_hidden1: delta_B_out = ", delta_B_out)
    #print("Jacobian_vector_for_hidden1: delta_w_out = ", delta_w_out)
    #print("\n")
    
    # Determine the delta vector of hidden neuron biases
    delta_B_HL1 = np.array([[(1-x)*pow(x,2) for x in sample] for sample in deepcopy(delta_w_out)])
    #print("Jacobian_vector_for_hidden1: delta_B_HL1 = ", delta_B_HL1)
    #print("\n")
    
    delta_w_HL1 = np.array([None for i in range(len(activated_hidden1_outputs))])
    for i in range(len(activated_hidden1_outputs)):
        delta_w_HL1[i] = delta_W_HL1_computation(activated_hidden1_outputs[i], )
    
    #delta_w_HL1 = 
        
    
    return Jacob_W_HL1_new

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
        
        Tr_calculated_outputs = feedforward_comp(W_HL1_new, B_HL1_new, W_out_new, B_out_new, Tr_data_inputs)
        Tr_errors, Tr_avg_error = error_checking(Tr_calculated_outputs, Tr_data_outputs)

        V_calculated_outputs = feedforward_comp(W_HL1_new, B_HL1_new, W_out_new, B_out_new, V_data_inputs)
        V_errors, V_avg_error = error_checking(V_calculated_outputs, V_data_outputs)
        
        if(Tr_avg_error < user_Validation_error or V_avg_error == Tr_avg_error):
            break
        
        activated_hidden1_outputs = activated_neuron_function(W_HL1_new, B_HL1_new, W_out_new, B_out_new, Tr_data_inputs)
        Jacob_W_HL1_new = Jacobian_vector_for_HL1(W_HL1_new, B_HL1_new, W_out_new, B_out_new, activated_hidden1_outputs, Tr_data_inputs, Tr_calculated_outputs, Tr_data_outputs)
        
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
