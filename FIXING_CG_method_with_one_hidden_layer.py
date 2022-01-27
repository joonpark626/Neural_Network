from copy import deepcopy
from operator import index
import numpy as np
from math import *
import pandas as pd
import csv
from random import random

#Transfer neuron activation sigmoid function
def transfer_sigmoid(activation):
    return 1.0/(1.0 + exp(-activation))

def forward_propagate(W_new_out, W_new_HLN1, data_input_Vgs, data_input_Vds):
    calculated_outputs = []
    HL1_activated_list = []
    HL1_to_Output_Neuron_list = []
    temp = 0
    
    #print("---------------------------------------------------------------")
    #print("forward_propagate: W_new_out = ", W_new_out)
    #print("forward_propagate: W_new_HLN1 = ", W_new_HLN1)
    #print("\n")
    
    for i in range(len(data_input_Vds)):
        for weights in W_new_HLN1:
            for element in range(len(weights)):
                if (element == 0):
                    temp = temp + weights[element]*data_input_Vgs[i]
                elif (element == 1):
                    temp = temp + weights[element]*data_input_Vds[i]
                else:
                    temp = temp + weights[element]
            HL1_activated_list.extend([transfer_sigmoid(temp)])
            temp = 0
        #print("forward_propagate: HL1_activated_list = ", HL1_activated_list)

        for j in range(len(W_new_out)):
            if (j != (len(W_new_out) - 1)):
                temp = temp + W_new_out[j]*HL1_activated_list[j]
            else:
                temp = temp + W_new_out[j]
        HL1_to_Output_Neuron_list.extend([temp])
        temp = 0
        
        calculated_outputs.extend(deepcopy(HL1_to_Output_Neuron_list))
        HL1_to_Output_Neuron_list.clear()
        HL1_activated_list.clear()
    
    #print("forward_propagate: calculated_outputs = ", calculated_outputs)
    #print("\n")
    return calculated_outputs

def forward_propagate_newWeights(W_new_out, W_new_HLN1, data_input_Vgs, data_input_Vds):
    
    #print("forward_propagate_newWeights: W_new_out = ", W_new_out)
    #print("forward_propagate_newWeights: W_new_HLN1 = ", W_new_HLN1)
    #print("\n")
    calculated_outputs = []
    HL1_activated_list = []
    HL1_to_Output_Neuron_list = []
    Activated_HLN1_list_for_DE = []
    
    temp = 0
    for i in range(len(data_input_Vds)):
        for weights in W_new_HLN1:
            for element in range(len(weights)):
                if (element == 0):
                    temp = temp + weights[element]*data_input_Vgs[i]
                elif (element == 1):
                    temp = temp + weights[element]*data_input_Vds[i]
                else:
                    temp = temp + weights[element]
            HL1_activated_list.extend([transfer_sigmoid(temp)])
            
            temp = 0
        #print("forward_propagate_newWeights: HL1_activated_list = ", HL1_activated_list)
        
        Activated_HLN1_list_for_DE.append(deepcopy(HL1_activated_list))
        for j in range(len(W_new_out)):
            if (j != (len(W_new_out) - 1)):
                temp = temp + W_new_out[j]*HL1_activated_list[j]
            temp = temp + W_new_out[-1]
        HL1_to_Output_Neuron_list.extend([temp])
        temp = 0
        
        calculated_outputs.extend(deepcopy(HL1_to_Output_Neuron_list))
        HL1_to_Output_Neuron_list.clear()
        HL1_activated_list.clear()
    print("\n")
    
    # print("forward_propagate: calculated_outputs = ", calculated_outputs)
    # print("\n")
    # print("forward_propagate: Activated_HLN1_list_for_DE \n= ", np.array(Activated_HLN1_list_for_DE))
    # print("\n")
    return calculated_outputs, Activated_HLN1_list_for_DE

def Phi_function(W_out_original, W_hidden_original, h_out_original, h_hidden_original, data_input_Vgs, data_input_Vds, data_outputs, etah):
    
    # About: This function is for calculating an error with the etah value from line minimization  

    delta_W_out = np.dot(h_out_original, etah)
    delta_W_hidden = np.dot(h_hidden_original, etah)
    
    W_out_phi = np.array(W_out_original) + np.array(delta_W_out)
    W_hidden_phi = np.array(W_hidden_original) + np.array(delta_W_hidden)

    
    calculated_outputs = forward_propagate(W_out_phi, W_hidden_phi, data_input_Vgs, data_input_Vds)
    phi = error_checking(calculated_outputs, data_outputs)
    #print("Phi_function: phi = ", phi)    
    
    return phi

def Generate_DE_Hidden_Neuron_each_sample(Activated_HLN1_list_for_DE, data_input_Vgs, data_input_Vds):
    
    # About: using each training data sample, create and compute the derivative of weights in hidden layer in nested list.
    # Ex) 3 Neurons => [[1], [2], [3]]

    temp = []
    temp_sample = []
    for weight in Activated_HLN1_list_for_DE:
        temp.extend([pow(weight,2)*(1 - weight)*data_input_Vgs])
        temp.extend([pow(weight,2)*(1 - weight)*data_input_Vds])
        temp.extend([pow(weight,2)*(1 - weight)])
        temp_sample.append(deepcopy(temp))
        temp.clear()
    Epoch_DE_hidden_neuron_vector = temp_sample
    
    return Epoch_DE_hidden_neuron_vector


# Comparing the calculated and measured outputs for conjugate gradient iteration
def error_checking(calculated_outputs, data_outputs):
    
    error = 0
    for i in range(len(calculated_outputs)):
        error += 0.5*(pow(calculated_outputs[i] - data_outputs[i], 2))
    return error


def delta_E_function(W_new_out, W_new_HLN1, data_input_Vgs, data_input_Vds, data_outputs):

    # About: With the present epoch weights, derivative of output weight and hidden weight will be determined
    # State the derivate of a function for each neuron weight/bias
    
    calculated_outputs, Activated_HLN1_list_for_DE = forward_propagate_newWeights(W_new_out, W_new_HLN1, data_input_Vgs, data_input_Vds)

    Diff_Calc_Measure = []
    for i in range(len(data_input_Vds)):
        Diff_Calc_Measure.extend([calculated_outputs[i] - data_outputs[i]])
    
    # -------------------------------- Output neuron below ----------------------------------------    
    DE_Output_neuron_vector = []
    temp_Activated_HLN1_list_for_DE = deepcopy(Activated_HLN1_list_for_DE)
    for i in range(len(temp_Activated_HLN1_list_for_DE)):
        temp_Activated_HLN1_list_for_DE[i].append(1)
        DE_Output_neuron_vector.append(np.dot(temp_Activated_HLN1_list_for_DE[i], Diff_Calc_Measure[i]))
    
    SUM_DE_Output_neuron_vector = [sum(i) for i in zip(*DE_Output_neuron_vector)]
    SUM_neg_DE_Output_neuron_vector = [-x for x in deepcopy(SUM_DE_Output_neuron_vector)]

    # -------------------------------- Hidden neurons below ----------------------------------------
    # Delta E vector for the weights and the bias of the output neuron with entire data set(12 in total)
    # Prev_#### means only the vector portion, not with the multiplication of (calc-measure)
    
    # Create 12 nested lists == 12 samples, each nested list contains derivative of weights for each neuron
    Epoch_DE_hidden_neuron_vector = [None for i in range(len(Activated_HLN1_list_for_DE))]  
    for i in range(len(Activated_HLN1_list_for_DE)):
        Epoch_DE_hidden_neuron_vector[i] = Generate_DE_Hidden_Neuron_each_sample(Activated_HLN1_list_for_DE[i], data_input_Vgs[i], data_input_Vds[i])

    # Multiplying by the 12 differences (calculated - measured) for respective DE matrix frmo Epoch_DE_hidden_neuron_vector
    # SUM delta E = (calc - measu)*
    MUL_Epoch_DE_hidden_neuron_vector = [None for i in range(len(Activated_HLN1_list_for_DE))]
    for i in range(len(Epoch_DE_hidden_neuron_vector)):
        MUL_Epoch_DE_hidden_neuron_vector[i] = np.dot(Epoch_DE_hidden_neuron_vector[i], Diff_Calc_Measure[i])
    
    # There are 12 matrices (12 DE vectors from 12 data samples) and sum all and obtain one hidden DE matrix
    SUM_DE_hidden_neuron_vector = np.zeros(np.shape(MUL_Epoch_DE_hidden_neuron_vector[0]))
    for i in range(len(MUL_Epoch_DE_hidden_neuron_vector)):
        SUM_DE_hidden_neuron_vector = SUM_DE_hidden_neuron_vector + np.array(MUL_Epoch_DE_hidden_neuron_vector[i])
    
    SUM_neg_DE_hidden_neuron_vector = [-x for x in deepcopy(SUM_DE_hidden_neuron_vector)]
                   
    return SUM_DE_Output_neuron_vector, SUM_neg_DE_Output_neuron_vector, SUM_DE_hidden_neuron_vector, SUM_neg_DE_hidden_neuron_vector  

def gamma_function(DE_Output_neuron_vector, DE_hidden_neuron_vector):
    
    #print("--------------------------------------------------------------------")
    # The length of DE_out and DE_hidden is 2
    # The first element of both is the present epoch and second is the previous epoch
    
    #print("gamma_function : DE_Output_neuron_vector = ", DE_Output_neuron_vector)
    #print("gamma_function : DE_hidden_neuron_vector = ", DE_hidden_neuron_vector)
    #print("\n")
    
    num = 0         # initialize the numerator and the denominator to zero
    denom = 0

    for i in range(len(DE_Output_neuron_vector)):
        if (i == 0):
            for neuron in DE_Output_neuron_vector[i]:
                num = num + pow(neuron,2)
                
            for neuron in DE_hidden_neuron_vector[i]:
                for j in range(len(neuron)):
                    num = num + pow(neuron[j],2)
                    
        else:
            for neuron in DE_Output_neuron_vector[i]:
                denom = denom + pow(neuron,2)
                
            for neuron in DE_hidden_neuron_vector[i]:
                for j in range(len(neuron)):
                    denom = denom + pow(neuron[j],2)
    
    #print("gamma_function : num = ", num)
    #print("gamma_function : denom = ", denom)
    #print("\n")
    
    temp_num = sqrt(num)
    temp_denom = sqrt(denom)
    gamma = temp_num/temp_denom
    
    # print("gamma_function : pow(num,2) = ", pow(num,2))
    # print("gamma_function : pow(denom,2) = ", pow(denom,2))
    print("gamma_function : gamma = ", gamma)
    # print("\n")
    
    return gamma

def line_search_function_h_hidden(W_new_out, W_new_HLN1, h_out, h_hidden, data_input_Vgs, data_input_Vds, data_outputs):
    # check the line search method from prof's textbook (pg.60~)
    # etah_1 = min(=0) , etah_2 = max(= fixed number, user given)4

    count = 0
    zero_threshold = 0.00000001 # seven zeros originally
    
    #step 1
    etah1 = 0
    etah2 = 0.5
    etah3 = etah2 - 0.618*(etah2-etah1)
    etah4 = etah1 + 0.618*(etah2-etah1)
    
    W_out_original = W_new_out
    W_hidden_original = W_new_HLN1
    h_out_original = h_out
    h_hidden_original = h_hidden
    
    # print("line_search_function_h_hidden: W_out_original : ", W_out_original)
    # print("line_search_function_h_hidden: W_hidden_original : ", W_hidden_original)
    # print("\n")
    # print("line_search_function_h_hidden: h_out_original : ", h_out_original)
    # print("line_search_function_h_hidden: h_hidden_original : ", h_hidden_original)
    # print("\n")
    
    #step2
    for i in range(1000):
        
        # Phi is the calculated error with W + etah*h
        phi3 = Phi_function(W_out_original, W_hidden_original, h_out_original, h_hidden_original, data_input_Vgs, data_input_Vds, data_outputs, etah3)
        #print("line_search_function_h_hidden: phi3 : ", phi3)
        
        phi4 = Phi_function(W_out_original, W_hidden_original, h_out_original, h_hidden_original, data_input_Vgs, data_input_Vds, data_outputs, etah4)
        #print("line_search_function_h_hidden: phi4 : ", phi4)
        #print("\n")
        
        #------------------------------------------------------------------#
        if (phi3 > phi4):
            etah1 = etah3
            etah3 = etah4
            etah4 = etah1 + 0.618*(etah2-etah1)

            phi4 = Phi_function(W_out_original, W_hidden_original, h_out_original, h_hidden_original, data_input_Vgs, data_input_Vds, data_outputs, etah4)
            
        else :
            etah2 = etah4
            etah4 = etah3
            etah3 = etah2 - 0.618*(etah2-etah1)
        
            phi3 = Phi_function(W_out_original, W_hidden_original, h_out_original, h_hidden_original, data_input_Vgs, data_input_Vds, data_outputs, etah3)
            
        if (abs(etah2 - etah1) < zero_threshold):
            etah = etah3
            break
        count = count + 1
              
    #print("line_search_function_h_hidden : count = ", count)
    print("line_search_function_h_hidden : etah = ", etah)
    print("\n")
    return etah

def conjugate_gradient(user_Validation_error, W_out, W_HLN1, data_Tr_input_Vgs, data_Tr_input_Vds, data_Tr_output_Id, data_V_input_Vgs, data_V_input_Vds, data_V_output_Id, data_Te_input_Vgs, data_Te_input_Vds, data_Te_output_Id, max_epoch):
    
    Training_error_collection = []
    num_epoch_iterated = 0
    W_new_out = [W_out, None]
    W_new_HLN1 = [W_HLN1, None]
    h_out = [None, None]
    h_hidden = [None, None]
    Delta_E_W_out  = [None, None]
    Delta_E_W_hidden  = [None, None]
    
    
    for epoch in range(max_epoch):
        #print("----------------------------------------------")
        #print("conjugate_gradient : epoch = ", epoch)
        
        calculated_outputs_Tr = forward_propagate(W_new_out[0], W_new_HLN1[0], data_Tr_input_Vgs, data_Tr_input_Vds)
        FF_error_Tr = error_checking(calculated_outputs_Tr, data_Tr_output_Id)
        
        calculated_outputs_V = forward_propagate(W_new_out[0], W_new_HLN1[0], data_V_input_Vgs, data_V_input_Vds)
        FF_error_V = error_checking(calculated_outputs_V, data_V_output_Id)
        
        #print("conjugate_gradient : calculated_outputs_Tr = ", calculated_outputs_Tr)
        #print("conjugate_gradient : FF_error_Tr = ", FF_error_Tr)
        #print("conjugate_gradient : FF_error_V = ", FF_error_V)
        #print("\n")
        
        Training_error_collection.extend([FF_error_Tr])
        #print("conjugate_gradient: FF_error_Tr = ", FF_error_Tr)
        if (FF_error_Tr < user_Validation_error):
            break

        DE_Output_neuron_vector, neg_DE_Output_neuron_vector, DE_hidden_neuron_vector, neg_DE_hidden_neuron_vector = delta_E_function(W_new_out[0], W_new_HLN1[0], data_Tr_input_Vgs, data_Tr_input_Vds, data_Tr_output_Id)
        Delta_E_W_out.insert(0, deepcopy(DE_Output_neuron_vector))
        Delta_E_W_hidden.insert(0, deepcopy(DE_hidden_neuron_vector))
        Delta_E_W_out.pop(-1)
        Delta_E_W_hidden.pop(-1)
        
        # print("conjugate_gradient: DE_Output_neuron_vector = ", DE_Output_neuron_vector)
        # print("conjugate_gradient: neg_DE_Output_neuron_vector = ", neg_DE_Output_neuron_vector)
        # print("conjugate_gradient: DE_hidden_neuron_vector = ", DE_hidden_neuron_vector)
        # print("conjugate_gradient: neg_DE_hidden_neuron_vector = ", neg_DE_hidden_neuron_vector)
        # print("\n")
        
        # print("conjugate_gradient: Delta_E_W_out = ", Delta_E_W_out)
        # print("conjugate_gradient: Delta_E_W_hidden = ", Delta_E_W_hidden)
        # print("\n")
    

        if (epoch == 0):
            
            W_new_out_epoch0 = deepcopy(W_new_out[0])
            W_new_HLN1_epoch0 = deepcopy(W_new_HLN1[0])
            
            h_out.insert(0, deepcopy(neg_DE_Output_neuron_vector))
            h_hidden.insert(0, deepcopy(neg_DE_hidden_neuron_vector))
            
            h_out.pop(-1)
            h_hidden.pop(-1)
            
            h_out_epoch0 = deepcopy(h_out[0])
            h_hidden_epoch0 = deepcopy(h_hidden[0])
            
            etah = line_search_function_h_hidden(W_new_out_epoch0, W_new_HLN1_epoch0, h_out_epoch0, h_hidden_epoch0, data_Tr_input_Vgs, data_Tr_input_Vds, data_Tr_output_Id)

            W_out_final_epoch0 = np.array(W_new_out_epoch0) + np.array(np.dot(h_out_epoch0, etah))
            W_hidden_final_epoch0 = np.array(W_new_HLN1_epoch0) + np.array(np.dot(h_hidden_epoch0, etah))
            #print("conjugate_gradient: W_out_final DEBUG = ", W_out_final)
            #print("conjugate_gradient: W_hidden_final DEBUG = ", W_hidden_final)
            #print("\n")
            
            W_new_out.insert(0, deepcopy(W_out_final_epoch0))
            W_new_HLN1.insert(0, deepcopy(W_hidden_final_epoch0))
            #print("conjugate_gradient: W_new_out = ", np.array(W_new_out))
            #print("conjugate_gradient: W_new_HLN1 = ", np.array(W_new_HLN1))
            #print("\n")
            
            W_new_out.pop(-1)
            W_new_HLN1.pop(-1)
            #print("conjugate_gradient: W_new_out DEBUG = ", np.array(W_new_out))
            #print("conjugate_gradient: W_new_HLN1 DEBUG = ", np.array(W_new_HLN1))
            #print("\n")
            
        else:
            #print("conjugate_gradient: DE_Output_neuron_vector (epoch >= ",epoch,") = ", DE_Output_neuron_vector)
            #print("conjugate_gradient: DE_hidden_neuron_vector (epoch >= ",epoch,") = ", DE_hidden_neuron_vector)
            #print("\n")
            
            #print("conjugate_gradient: Delta_E_W_out (epoch >= ",epoch,") = ", Delta_E_W_out)
            #print("conjugate_gradient: Delta_E_W_hidden (epoch >= ",epoch,") = ", Delta_E_W_hidden)
            #print("\n")
            
            gamma = gamma_function(Delta_E_W_out, Delta_E_W_hidden)
            #print("conjugate_gradient: gamma (epoch >= ",epoch,") = ", gamma)
            #print("\n")
            
            # print("conjugate_gradient: h_out (epoch >= ",epoch,") = ", h_out)
            # print("conjugate_gradient: h_hidden (epoch >= ",epoch,") = ", h_hidden)
            # print("conjugate_gradient: neg_DE_Output_neuron_vector (epoch >= ",epoch,") = ", neg_DE_Output_neuron_vector)
            # print("conjugate_gradient: neg_DE_hidden_neuron_vector (epoch >= ",epoch,") = ", neg_DE_hidden_neuron_vector)
            # print("\n")
            
            h_out.pop(-1)
            h_hidden.pop(-1)
            
            etah = line_search_function_h_hidden(W_new_out[0], W_new_HLN1[0], h_out[0], h_hidden[0], data_Tr_input_Vgs, data_Tr_input_Vds, data_Tr_output_Id)
            
            #print("conjugate_gradient: h_out (epoch >= ",epoch,") = ", h_out)
            #print("conjugate_gradient: h_hidden (epoch >= ",epoch,") = ", h_hidden)
            #print("\n")
            
            # h(epoch - 1)
            delta_h_out = np.dot(h_out[0], gamma)
            delta_h_hidden = np.dot(h_hidden[0], gamma)
            
            #print("conjugate_gradient: delta_h_out (epoch >= ",epoch,") DEBUG = ", delta_h_out)
            #print("conjugate_gradient: delta_h_hidden (epoch >= ",epoch,") DEBUG = ", delta_h_hidden)
            #print("\n")
            
            final_h_out = np.array(neg_DE_Output_neuron_vector) + np.array(delta_h_out)
            final_h_hidden = np.array(neg_DE_hidden_neuron_vector) + np.array(delta_h_hidden)
            
            #print("conjugate_gradient: final_h_out (epoch >= ",epoch,") = ", final_h_out)
            #print("conjugate_gradient: final_h_hidden (epoch >= ",epoch,") = ", final_h_hidden)
            #print("\n")
            
            h_out.insert(0,deepcopy(final_h_out))
            h_hidden.insert(0,deepcopy(final_h_hidden))
            #print("conjugate_gradient: h_out (epoch >= ",epoch,") = ", h_out)
            #print("conjugate_gradient: h_hidden (epoch >= ",epoch,") = ", h_hidden)
            #print("\n")
            
            W_out_final= np.array(W_new_out[0]) + np.array(np.dot(h_out[1], etah))
            W_hidden_final = np.array(W_new_HLN1[0]) + np.array(np.dot(h_hidden[1], etah))
            #print("conjugate_gradient: W_new_out (epoch >= ",epoch,") = ", W_new_out)
            #print("conjugate_gradient: W_new_HLN1 (epoch >= ",epoch,") = ", W_new_HLN1)
            #print("\n")
            
            W_new_out.insert(0, deepcopy(W_out_final))
            W_new_HLN1.insert(0, deepcopy(W_hidden_final))
            #print("conjugate_gradient: W_new_out (epoch >= ",epoch,") DEBUG1 = ", W_new_out)
            #print("conjugate_gradient: W_new_HLN1 (epoch >= ",epoch,") DEBUG1 = ", W_new_HLN1)
            #print("\n")
            
            W_new_out.pop(-1)
            W_new_HLN1.pop(-1)
            #print("conjugate_gradient: W_new_out (epoch >= ",epoch,") DEBUG2 = ", W_new_out)
            #print("conjugate_gradient: W_new_HLN1 (epoch >= ",epoch,") DEBUG2 = ", W_new_HLN1)
            #print("\n")

        num_epoch_iterated = num_epoch_iterated + 1
        #print("conjugate_gradient: num_epoch_iterated (epoch >= ",epoch,") = ", num_epoch_iterated)    
        calculated_outputs_Te = forward_propagate(W_new_out[0], W_new_HLN1[0], data_Te_input_Vgs, data_Te_input_Vds)
        FF_error_Te = error_checking(calculated_outputs_Te, data_Te_output_Id)

    return W_new_out[0], W_new_HLN1[0], num_epoch_iterated, Training_error_collection, FF_error_Te

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

# ---------------------------------------------------
# Data Generation and scaling of Training set
data_Tr_input_Vgs = []
data_Tr_input_Vds = []
temp_data_Tr_output_Id = []
for sample in data_Tr:
    for j in range(len(sample)):
        if (j == (len(sample)-3)):
            data_Tr_input_Vgs.append(sample[j])
        if (j == (len(sample)-2)):
            data_Tr_input_Vds.append(sample[j])
        if (j == (len(sample)-1)):
            temp_data_Tr_output_Id.append(sample[j])

# ---------------------------------------------------
# Data Generation and scaling of Validation set
data_V_input_Vgs = []
data_V_input_Vds = []
temp_data_V_output_Id = []
for sample2 in data_V:
    for j in range(len(sample2)):
        if (j == (len(sample2)-3)):
            data_V_input_Vgs.append(sample2[j])
        if (j == (len(sample2)-2)):
            data_V_input_Vds.append(sample2[j])
        if (j == (len(sample2)-1)):
            temp_data_V_output_Id.append(sample2[j])

# ---------------------------------------------------
# Data Generation and scaling of Validation set
data_Te_input_Vgs = []
data_Te_input_Vds = []
temp_data_Te_output_Id = []
for sample3 in data_Te:
    for j in range(len(sample3)):
        if (j == (len(sample3)-3)):
            data_Te_input_Vgs.append(sample3[j])
        if (j == (len(sample3)-2)):
            data_Te_input_Vds.append(sample3[j])
        if (j == (len(sample3)-1)):
            temp_data_Te_output_Id.append(sample3[j])

Tr_Id_min = min(temp_data_Tr_output_Id)
V_Id_min = min(temp_data_V_output_Id)
Te_Id_min = min(temp_data_Te_output_Id)

#------------------------------------------------
# Data Scaling in log (e)
scaled_Tr_output_Id = [None for i in range(len(temp_data_Te_output_Id))]
scaled_V_output_Id = [None for i in range(len(temp_data_Te_output_Id))]
scaled_Te_output_Id = [None for i in range(len(temp_data_Te_output_Id))]
for i in range(len(temp_data_Tr_output_Id)):
    if (temp_data_Tr_output_Id[i] == Tr_Id_min):
        scaled_Tr_output_Id[i] = 0
    else:
        scaled_Tr_output_Id[i] = log(temp_data_Tr_output_Id[i] - Tr_Id_min)
    if (temp_data_V_output_Id[i] == V_Id_min):
        scaled_V_output_Id[i] = 0
    else:
        scaled_V_output_Id[i] = log(temp_data_V_output_Id[i] - V_Id_min)
    if (temp_data_Te_output_Id[i] == Te_Id_min):
        scaled_Te_output_Id[i] = 0
    else:
        scaled_Te_output_Id[i] = log(temp_data_Te_output_Id[i] - Te_Id_min)    

print("scaled_Tr_output_Id : " , scaled_Tr_output_Id)
print("scaled_V_output_Id : " , scaled_V_output_Id)
print("scaled_Te_output_Id : " , scaled_Te_output_Id)
print("\n")

Ali_input_num = 2
temp_input = Ali_input_num

# Initialize the weight values when a number was typed by an user in GUI
hidden1 = 5
if (hidden1 > 0):              
    W_HLN1 = [[2*random()-1 for i in range(temp_input + 1)] for i in range(hidden1)] # Initializing random weights and biases in a range of -1 to 1
print("temp_W_HLN1 : " , W_HLN1)

W_out = [2*random()-1 for i in range(len(W_HLN1) + 1) ]

max_epoch = 20;
user_Validation_error = 0.1;
[W_new_out, W_new_HLN1, num_epoch_iter, Training_error_collection, FF_error_Te] = conjugate_gradient(user_Validation_error, W_out, W_HLN1, data_Tr_input_Vgs, data_Tr_input_Vds, scaled_Tr_output_Id, 
                                                                                                     data_V_input_Vgs, data_V_input_Vds, scaled_V_output_Id, data_Te_input_Vgs, data_Te_input_Vds, scaled_Te_output_Id, max_epoch)

print("W_new_out : " , W_new_out)
print("W_new_HLN1 : " , W_new_HLN1)
print("num_epoch_iter : " , num_epoch_iter)
print("Training_error_collection : " , Training_error_collection)
print("FF_error_Te : " , FF_error_Te)

#-----------------------NOTE----------------------------
#The structure of the ANN with one hidden layer
#                    Id
#        # of Hidden Neurons here
#                Vgs     Vds






