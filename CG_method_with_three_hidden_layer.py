from copy import deepcopy
import numpy as np
from math import *
import pandas as pd
import csv
from random import random

#Transfer neuron activation sigmoid function
def transfer_sigmoid(activation):
    return 1.0/(1.0 + exp(-activation))

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#---------------------- 3 Hidden Layer Functions Collection--------------------------

def forward_propagate_3HL(W_new_out, W_new_HLN3, W_new_HLN2, W_new_HLN1, data_input_Vgs, data_input_Vds):
    calculated_outputs = []
    HL1_activated_list = []
    HL2_activated_list = []
    HL3_activated_list = []
    HL3_to_Output_Neuron_list = []
    epoch_HL3_activated_list = []
    epoch_HL2_activated_list = []
    epoch_HL1_activated_list = []
    temp = 0
    
    # print("---------------------------------------------------------------")
    # print("forward_propagate: W_new_out = ", W_new_out)
    # print("forward_propagate: W_new_HLN3 = ", W_new_HLN3)
    # print("forward_propagate: W_new_HLN2 = ", W_new_HLN2)
    # print("forward_propagate: W_new_HLN1 = ", W_new_HLN1)
    # print("\n")
    
    for i in range(len(data_input_Vds)):
        for weights in W_new_HLN1:
            for element in range(len(weights)):
                if (element == 0):
                    temp = temp + weights[element]*data_input_Vgs[i]
                elif (element == 1):
                    temp = temp + weights[element]*data_input_Vds[i]
                else:
                    temp = temp + weights[element]
            HL1_activated_list.append(transfer_sigmoid(temp))
            epoch_HL1_activated_list.append(deepcopy(HL1_activated_list))
            temp = 0
        #print("forward_propagate: HL1_activated_list = ", HL1_activated_list)

        for weights2 in W_new_HLN2:
            for element2 in range(len(weights2)):
                if (element2 != (len(weights2) - 1)):
                    temp = temp + weights2[element2]*HL1_activated_list[element2]
                else:
                    temp = temp + weights2[element2]
            HL2_activated_list.append(transfer_sigmoid(temp))
            epoch_HL2_activated_list.append(deepcopy(HL2_activated_list))
            temp = 0
        #print("forward_propagate: HL2_activated_list = ", HL2_activated_list)
        
        for weights3 in W_new_HLN3:
            for element3 in range(len(weights3)):
                if (element3 != (len(weights3) - 1)):
                    temp = temp + weights3[element3]*HL1_activated_list[element3]
                else:
                    temp = temp + weights3[element3]
            HL3_activated_list.append(transfer_sigmoid(temp))
            epoch_HL3_activated_list.append(deepcopy(HL3_activated_list))
            temp = 0
        #print("forward_propagate: HL3_activated_list = ", HL3_activated_list)
        #print("\n")

        for j in range(len(W_new_out)):
            if (j != (len(W_new_out) - 1)):
                temp = temp + W_new_out[j]*HL3_activated_list[j]
            else:
                temp = temp + W_new_out[j]
        HL3_to_Output_Neuron_list.extend([temp])
        temp = 0
        
        calculated_outputs.extend(deepcopy(HL3_to_Output_Neuron_list))
        HL3_to_Output_Neuron_list.clear()
        HL3_activated_list.clear()
        HL2_activated_list.clear()
        HL1_activated_list.clear()
    
    #print("forward_propagate: calculated_outputs = ", calculated_outputs)
    #print("\n")    
    return calculated_outputs

def forward_propagate_3HL_with_Activated(W_new_out, W_new_HLN3, W_new_HLN2, W_new_HLN1, data_input_Vgs, data_input_Vds):
    calculated_outputs = []
    HL1_activated_list = []
    HL2_activated_list = []
    HL3_activated_list = []
    HL3_to_Output_Neuron_list = []
    epoch_HL3_activated_list = []
    epoch_HL2_activated_list = []
    epoch_HL1_activated_list = []
    temp = 0
    
    # print("---------------------------------------------------------------")
    # print("forward_propagate: W_new_out = ", W_new_out)
    # print("forward_propagate: W_new_HLN3 = ", W_new_HLN3)
    # print("forward_propagate: W_new_HLN2 = ", W_new_HLN2)
    # print("forward_propagate: W_new_HLN1 = ", W_new_HLN1)
    # print("\n")
    
    for i in range(len(data_input_Vds)):
        for weights in W_new_HLN1:
            for element in range(len(weights)):
                if (element == 0):
                    temp = temp + weights[element]*data_input_Vgs[i]
                elif (element == 1):
                    temp = temp + weights[element]*data_input_Vds[i]
                else:
                    temp = temp + weights[element]
            HL1_activated_list.append(transfer_sigmoid(temp))
            temp = 0
        epoch_HL1_activated_list.append(deepcopy(HL1_activated_list))
        #print("forward_propagate: HL1_activated_list = ", HL1_activated_list)

        for weights2 in W_new_HLN2:
            for element2 in range(len(weights2)):
                if (element2 != (len(weights2) - 1)):
                    temp = temp + weights2[element2]*HL1_activated_list[element2]
                else:
                    temp = temp + weights2[element2]
            HL2_activated_list.append(transfer_sigmoid(temp))
            temp = 0
        epoch_HL2_activated_list.append(deepcopy(HL2_activated_list))
        #print("forward_propagate: HL2_activated_list = ", HL2_activated_list)
        
        for weights3 in W_new_HLN3:
            for element3 in range(len(weights3)):
                if (element3 != (len(weights3) - 1)):
                    temp = temp + weights3[element3]*HL1_activated_list[element3]
                else:
                    temp = temp + weights3[element3]
            HL3_activated_list.append(transfer_sigmoid(temp))
            temp = 0
        epoch_HL3_activated_list.append(deepcopy(HL3_activated_list))    
        #print("forward_propagate: HL3_activated_list = ", HL3_activated_list)
        #print("\n")

        for j in range(len(W_new_out)):
            if (j != (len(W_new_out) - 1)):
                temp = temp + W_new_out[j]*HL3_activated_list[j]
            else:
                temp = temp + W_new_out[j]
        HL3_to_Output_Neuron_list.extend([transfer_sigmoid(temp)])
        temp = 0
        
        calculated_outputs.extend(deepcopy(HL3_to_Output_Neuron_list))
        HL3_to_Output_Neuron_list.clear()
        HL3_activated_list.clear()
        HL2_activated_list.clear()
        HL1_activated_list.clear()
    
    #print("forward_propagate: calculated_outputs = ", calculated_outputs)
    #print("\n")    
    return calculated_outputs, epoch_HL3_activated_list, epoch_HL2_activated_list,epoch_HL1_activated_list

# Comparing the calculated and measured outputs for conjugate gradient iteration
def error_checking(calculated_outputs, data_outputs):
    # print("---------------------------------------------------------------")
    # print("error_checking : W_new_out = ", W_new_out)
    # print("error_checking : W_new_HLN3 = ", W_new_HLN3)
    # print("error_checking : W_new_HLN2 = ", W_new_HLN2)
    # print("error_checking : W_new_HLN1 = ", W_new_HLN1)
    # print("\n")
    # print("error_checking : data_input_Vgs = ", data_input_Vgs)
    # print("error_checking : data_input_Vds = ", data_input_Vds)
    # print("\n")
    #print("error_checking : calculated_outputs = ", calculated_outputs)
    
    error = 0
    for i in range(len(calculated_outputs)):
        error += 0.5*(pow(calculated_outputs[i] - data_outputs[i], 2))
    #print("error_checking : error = ", error)
    return error

# Comparing the calculated and measured outputs for conjugate gradient iteration
def error_checking_Test_set(calculated_outputs, data_outputs):
    
    Test_error_collection = np.square(np.array(calculated_outputs) - np.array(data_outputs))*0.5
    
    return Test_error_collection

def Prep_DE_hidden3_vector_per_sample(Activated_HLN3_list_for_DE, Activated_HLN2_list_for_DE):
    # print("-----------------------------------------------------------------")
    # print("Prep_DE_hidden3_vector_per_sample: Activated_HLN3_list_for_DE : ", Activated_HLN3_list_for_DE)
    # print("Prep_DE_hidden3_vector_per_sample: Activated_HLN2_list_for_DE : ", Activated_HLN2_list_for_DE)
    # print("\n")
    
    temp_Activated_HLN3_list_for_DE = deepcopy(Activated_HLN3_list_for_DE)
    temp_Activated_HLN3_list_for_DE.pop(-1)
    
    # About: Determining the derivative of the weights/bias of the hidden 2 neurons
    temp = []
    DE_h3_neurons_vector = []
    for weight in temp_Activated_HLN3_list_for_DE:
        for i in range(len(temp_Activated_HLN3_list_for_DE)):
            temp.extend([pow(weight,2)*(1 - weight)*Activated_HLN2_list_for_DE[i]])         # Finding the derivative of each neurons in hidden layer 3
        temp.extend([pow(weight,2)*(1 - weight)])
        DE_h3_neurons_vector.append(deepcopy(temp))
        temp.clear()
    #print("Prep_DE_hidden3_vector_per_sample: DE_h3_neurons_vector : ", DE_h3_neurons_vector)
    #print("\n")

    return DE_h3_neurons_vector

def Prep_DE_hidden2_vector_per_sample(Activated_HLN3_list_for_DE, Activated_HLN2_list_for_DE, Activated_HLN1_list_for_DE):
    # print("-----------------------------------------------------------------")
    # print("Prep_DE_hidden2_vector_per_sample: Activated_HLN3_list_for_DE : ", Activated_HLN3_list_for_DE)
    # print("Prep_DE_hidden2_vector_per_sample: Activated_HLN2_list_for_DE : ", Activated_HLN2_list_for_DE)
    # print("Prep_DE_hidden2_vector_per_sample: Activated_HLN1_list_for_DE : ", Activated_HLN1_list_for_DE)
    # print("\n")
    
    temp_Activated_HLN3_list_for_DE = deepcopy(Activated_HLN3_list_for_DE)
    temp_Activated_HLN3_list_for_DE.pop(-1)
    
    # All the derivative values from hidden layer 3 neurons needs to be summed for each hidden 2 layer neuron
    SUM_DE_HL3_neurons = 0 
    for i in range(len(temp_Activated_HLN3_list_for_DE)):
        SUM_DE_HL3_neurons = SUM_DE_HL3_neurons + pow(deepcopy(Activated_HLN3_list_for_DE[i]),2)*(1 - Activated_HLN3_list_for_DE[i])
    #print("Prep_DE_hidden2_vector_per_sample: SUM_DE_HL3_neurons : ", SUM_DE_HL3_neurons)
    #print("\n")
    
    # Creating a list for each hidden 2 layer neurons that accummulated derived values from upper neurons (HL3 and output) by chain rule 
    DE_HL2_and_3_neurons = []
    for i in range(len(Activated_HLN2_list_for_DE)):
        DE_HL2_and_3_neurons.append(SUM_DE_HL3_neurons*pow(deepcopy(Activated_HLN2_list_for_DE[i]),2)*(1 - Activated_HLN2_list_for_DE[i]))
    #print("Prep_DE_hidden2_vector_per_sample: DE_HL2_and_3_neurons : ", DE_HL2_and_3_neurons)
    #print("\n")
    
    temp = []
    temp_neuron = []
    Prep_DE_h2_neurons_vector = []
    for neuron in deepcopy(DE_HL2_and_3_neurons):
        for i in range(len(Activated_HLN1_list_for_DE)):
            temp.extend([Activated_HLN1_list_for_DE[i]*neuron])
        temp.extend([SUM_DE_HL3_neurons])
        temp_neuron.extend(deepcopy(temp))
        temp.clear()
        Prep_DE_h2_neurons_vector.append(deepcopy(temp_neuron))
        temp_neuron.clear()
    #print("Prep_DE_hidden2_vector_per_sample: Prep_DE_h2_neurons_vector : ", Prep_DE_h2_neurons_vector)
            
    return Prep_DE_h2_neurons_vector

def Prep_DE_hidden1_vector_per_sample(Activated_HLN3_list_for_DE, Activated_HLN2_list_for_DE, Activated_HLN1_list_for_DE, data_input_Vgs, data_input_Vds):
    # print("-----------------------------------------------------------------")
    # print("Prep_DE_hidden1_vector_per_sample: Activated_HLN3_list_for_DE : ", Activated_HLN3_list_for_DE)
    # print("Prep_DE_hidden1_vector_per_sample: Activated_HLN2_list_for_DE : ", Activated_HLN2_list_for_DE)
    # print("Prep_DE_hidden1_vector_per_sample: Activated_HLN1_list_for_DE : ", Activated_HLN1_list_for_DE)
    # print("\n")
    # print("Prep_DE_hidden1_vector_per_sample: data_input_Vgs : ", data_input_Vgs)
    # print("Prep_DE_hidden1_vector_per_sample: data_input_Vds : ", data_input_Vds)
    # print("\n")
    
    temp_Activated_HLN1_list_for_DE = deepcopy(Activated_HLN1_list_for_DE)
    temp_Activated_HLN1_list_for_DE.pop(-1)
    
    # All the derivative values from hidden layer 3 neurons needs to be summed for each hidden 2 layer neuron
    SUM_DE_HL3_neurons = 0 
    for i in range(len(temp_Activated_HLN1_list_for_DE)):
        SUM_DE_HL3_neurons = SUM_DE_HL3_neurons + pow(deepcopy(Activated_HLN3_list_for_DE[i]),2)*(1 - Activated_HLN3_list_for_DE[i])
    #print("Prep_DE_hidden1_vector_per_sample: SUM_DE_HL3_neurons : ", SUM_DE_HL3_neurons)
    #print("\n")
    
    # All the derivative values from hidden layer 2 neurons needs to be summed for each hidden 1 layer neuron
    SUM_DE_HL2_neurons = 0 
    for i in range(len(temp_Activated_HLN1_list_for_DE)):
        SUM_DE_HL2_neurons = SUM_DE_HL2_neurons + pow(deepcopy(Activated_HLN2_list_for_DE[i]),2)*(1 - Activated_HLN2_list_for_DE[i])
    #print("Prep_DE_hidden1_vector_per_sample: SUM_DE_HL3_neurons : ", SUM_DE_HL3_neurons)
    #print("\n")
    
    # Total accummulated DE values from HL 2 and 3 by chain rule
    sum_delta_from_HL23 = SUM_DE_HL3_neurons*SUM_DE_HL2_neurons
    #print("Prep_DE_hidden1_vector_per_sample: sum_delta_from_HL23 : ", sum_delta_from_HL23)
    #print("\n")
    
    temp = []    
    Prep_DE_hidden1_vector = []
    for i in range(len(Activated_HLN1_list_for_DE)):
        temp.extend([pow(Activated_HLN1_list_for_DE[i],2)*(1 - Activated_HLN1_list_for_DE[i])*sum_delta_from_HL23*data_input_Vgs])
        temp.extend([pow(Activated_HLN1_list_for_DE[i],2)*(1 - Activated_HLN1_list_for_DE[i])*sum_delta_from_HL23*data_input_Vds])
        temp.extend([pow(Activated_HLN1_list_for_DE[i],2)*(1 - Activated_HLN1_list_for_DE[i])*sum_delta_from_HL23])
        Prep_DE_hidden1_vector.append(deepcopy(temp))
        temp.clear()
    #print("Prep_DE_hidden1_vector_per_sample: Prep_DE_hidden1_vector DEBUG : ", np.array(Prep_DE_hidden1_vector))
    
    return Prep_DE_hidden1_vector

def W_or_DE_Mul_with_etah(W_out_original, W_hidden3_original, W_hidden2_original, W_hidden1_original, h_out_original, h_hidden3_original, h_hidden2_original, h_hidden1_original, etah):

    # About: h vectors need to be multiplied by the etah (becomes delta W) 
    # and then add with the current = original W vectors, W = W + delta_W 
        
    # Delta W for out, hidden2 and hidden 1 are ready to be added with W present
    delta_W_out = [x*etah for x in deepcopy(h_out_original)]
    delta_W_hidden3 = [[x*etah for x in neuron] for neuron in deepcopy(h_hidden3_original)]
    delta_W_hidden2 = [[x*etah for x in neuron] for neuron in deepcopy(h_hidden2_original)]
    delta_W_hidden1 = [[x*etah for x in neuron] for neuron in deepcopy(h_hidden1_original)]
    
    # Create a new vector for phi function shown in the line_search function
    W_out_phi = np.array(W_out_original) + np.array(delta_W_out)
    #print("W_or_DE_Mul_with_etah : W_out_phi = ", W_out_phi)
    
    W_hidden3_phi = np.array(W_hidden3_original) + np.array(delta_W_hidden3)
    #print("W_or_DE_Mul_with_etah : W_hidden3_phi = ", W_hidden3_phi)
    
    W_hidden2_phi = np.array(W_hidden2_original) + np.array(delta_W_hidden2)
    #print("W_or_DE_Mul_with_etah : W_hidden2_phi = ", W_hidden2_phi)
    
    W_hidden1_phi = np.array(W_hidden1_original) + np.array(delta_W_hidden1)
    #print("W_or_DE_Mul_with_etah : W_hidden1_phi = ", W_hidden1_phi)
    #print("\n")
    
    return W_out_phi, W_hidden3_phi, W_hidden2_phi, W_hidden1_phi

def function_for_phi_calculation(W_out_original, W_hidden3_original, W_hidden2_original, W_hidden1_original, h_out_original, h_hidden3_original, h_hidden2_original, h_hidden1_original, data_input_Vgs, data_input_Vds, data_outputs, etah):
    
    # About: For phi function of line minimization, phi(etah) = E(w + etah*h), "w + etah*h" needs to be determined by the function call below, with new W vector for phi,
    #        implement FF computation to measure the phi (= error)
    W_out_phi, W_hidden3_phi, W_hidden2_phi, W_hidden1_phi = W_or_DE_Mul_with_etah(W_out_original, W_hidden3_original, W_hidden2_original, W_hidden1_original, h_out_original, h_hidden3_original, h_hidden2_original, h_hidden1_original, etah)
    #print("line_search_function_h_hidden: W_hidden_for_phi : ", W_hidden_for_phi)
    
    calculated_outputs = forward_propagate_3HL(W_out_phi, W_hidden3_phi, W_hidden2_phi, W_hidden1_phi, data_input_Vgs, data_input_Vds)
    phi = error_checking(calculated_outputs, data_outputs)
    #print("function_for_phi_calculation: calculated_outputs : ", calculated_outputs)
    #print("function_for_phi_calculation: phi : ", phi)
    
    return phi

def hidden_neuron_allocation(Prep_DE_hidden_neuron_vector, index):
    
    #print("------------------------------------------------------------------")
    # About: Index = position of the neuron within the hidden layer 2
    #        Need to classify them for summation
    neuron_with_epoch = []
    for i in range(len(Prep_DE_hidden_neuron_vector)):
        for j in range(len(Prep_DE_hidden_neuron_vector[i])):
            if (index == j):
                neuron_with_epoch.append(deepcopy(Prep_DE_hidden_neuron_vector[i][j]))
    
    return neuron_with_epoch

def DE_out_and_neg_DE_out_vector(Activated_HLN_list_for_DE, Diff_Calc_Measure):
    
    # -------------------------------- Output neuron below ----------------------------------------
    # Can be reused for all the different hidden layers of conjugate methods
    # Prev_DE... vector is the collection of the derivatives of the weights/bias
    
    # The derivative of the output neuron with 12 data samples (Not yet, Multiplying with Diff_Calc_Measure)
    Prev_DE_Output_neuron_vector = []
    for sample in Activated_HLN_list_for_DE:
        sample.extend([1])
        Prev_DE_Output_neuron_vector.append(sample)
        
    # the derivative of the output neuron with 12 data samples (Yes, Multiplying with Diff_Calc_Measure)
    for element in Prev_DE_Output_neuron_vector:
        for i in range(len(element)):
            element[i] = element[i]*Diff_Calc_Measure[i]
            
    DE_Output_neuron_vector = deepcopy(Prev_DE_Output_neuron_vector)
    
    return DE_Output_neuron_vector

def SUM_DE_hidden3_neuron_vector_func(Activated_HLN3_list_for_DE, Activated_HLN2_list_for_DE, Diff_Calc_Measure):
    
    # About: Find the delta E vector of 3rd hidden layer neuron weights, multiply the difference results of respective sample 
    
    temp_Prep_DE_hidden3_neuron_vector = [None for i in range(12)]
    
    for i in range(12):
        temp_Prep_DE_hidden3_neuron_vector[i] = Prep_DE_hidden3_vector_per_sample(Activated_HLN3_list_for_DE[i], Activated_HLN2_list_for_DE[i])
    # for i in range(12):
    #    print("SUM_DE_hidden3_neuron_vector_func : temp_Prep_DE_hidden3_neuron_vector[",i,"] = ", temp_Prep_DE_hidden3_neuron_vector[i])
    # print("\n")
    
    Prep_DE_hidden3_neuron_vector = deepcopy(temp_Prep_DE_hidden3_neuron_vector)
    # Multiplying with the Diff_Calc_measure for all the 12 samples
    for i in range(len(temp_Prep_DE_hidden3_neuron_vector)):
        Prep_DE_hidden3_neuron_vector[i] = np.dot(Prep_DE_hidden3_neuron_vector[i],Diff_Calc_Measure[i])
    
    # for i in range(12):
    #     print("SUM_DE_hidden3_neuron_vector_func : Prep_DE_hidden3_neuron_vector [",i,"] = ", Prep_DE_hidden3_neuron_vector[i])
    # print("\n")
    
    SUM_DE_hidden3_neuron_vector = deepcopy(Prep_DE_hidden3_neuron_vector[-1])
    for i in range(len(Prep_DE_hidden3_neuron_vector)-1):
        SUM_DE_hidden3_neuron_vector = Prep_DE_hidden3_neuron_vector[i] + SUM_DE_hidden3_neuron_vector
    #print("SUM_DE_hidden3_neuron_vector_func : SUM_DE_hidden3_neuron_vector = ", SUM_DE_hidden3_neuron_vector)
    #print("\n")

    return SUM_DE_hidden3_neuron_vector

def Func_Prep_DE_hidden2_neuron_vector(Activated_HLN3_list_for_DE, Activated_HLN2_list_for_DE, Activated_HLN1_list_for_DE, Diff_Calc_Measure):
    # print("Func_Prep_DE_hidden2_neuron_vector : Activated_HLN3_list_for_DE = ", Activated_HLN3_list_for_DE)
    # print("\n")
    # print("Func_Prep_DE_hidden2_neuron_vector : Activated_HLN2_list_for_DE = ", Activated_HLN2_list_for_DE)
    # print("\n")
    # print("Func_Prep_DE_hidden2_neuron_vector : Activated_HLN1_list_for_DE = ", Activated_HLN1_list_for_DE)
    # print("\n")
    
    Prep_DE_hidden2_neuron_vector = [None for i in range(12)]
    
    for i in range(12):
        Prep_DE_hidden2_neuron_vector[i] = Prep_DE_hidden2_vector_per_sample(Activated_HLN3_list_for_DE[i], Activated_HLN2_list_for_DE[i], Activated_HLN1_list_for_DE[i])
    print("\n")
    # for i in range(12):
    #     print("Func_Prep_DE_hidden2_neuron_vector : Prep_DE_hidden2_neuron_vector[",i,"] = ", Prep_DE_hidden2_neuron_vector[i])
    # print("\n")
    # print("Func_Prep_DE_hidden2_neuron_vector : Diff_Calc_Measure = ", Diff_Calc_Measure)
    # print("\n")
    
    temp_Prep_DE_hidden2_neuron_vector = deepcopy(Prep_DE_hidden2_neuron_vector)
    # Multiplying with the Diff_Calc_measure for all the 12 samples
    for i in range(len(Prep_DE_hidden2_neuron_vector)):
        temp_Prep_DE_hidden2_neuron_vector[i] = np.dot(temp_Prep_DE_hidden2_neuron_vector[i],Diff_Calc_Measure[i])
                
    return Prep_DE_hidden2_neuron_vector

def Func_Prep_DE_hidden1_neuron_vector(Activated_HLN3_list_for_DE, Activated_HLN2_list_for_DE, Activated_HLN1_list_for_DE, Diff_Calc_Measure, data_input_Vgs, data_input_Vds):
    
    # print("Func_Prep_DE_hidden1_neuron_vector : Activated_HLN3_list_for_DE = ", Activated_HLN3_list_for_DE)
    # print("\n")
    # print("Func_Prep_DE_hidden1_neuron_vector : Activated_HLN2_list_for_DE = ", Activated_HLN2_list_for_DE)
    # print("\n")
    # print("Func_Prep_DE_hidden1_neuron_vector : Activated_HLN1_list_for_DE = ", Activated_HLN1_list_for_DE)
    # print("\n")
    
    
    Prep_DE_hidden1_neuron_vector = [None for i in range(12)]
    for i in range(12):
        Prep_DE_hidden1_neuron_vector[i] = Prep_DE_hidden1_vector_per_sample(Activated_HLN3_list_for_DE[i], Activated_HLN2_list_for_DE[i], Activated_HLN1_list_for_DE[i],data_input_Vgs[i], data_input_Vds[i])
    #for i in range(12):
    #    print("Func_Prep_DE_hidden1_neuron_vector : Prep_DE_hidden1_neuron_vector[",i,"] = ", Prep_DE_hidden1_neuron_vector[i])
    #print("\n")
    
    # Below, multiplying by diff_calc_measure for each dta sample
    # Multiplying with the Diff_Calc_measure for all the 12 samples
    
    temp_Prep_DE_hidden1_neuron_vector = deepcopy(Prep_DE_hidden1_neuron_vector)
    # Multiplying with the Diff_Calc_measure for all the 12 samples
    for i in range(len(temp_Prep_DE_hidden1_neuron_vector)):
        temp_Prep_DE_hidden1_neuron_vector[i] = np.dot(temp_Prep_DE_hidden1_neuron_vector[i],Diff_Calc_Measure[i])
    #for i in range(12):
    #    print("Func_Prep_DE_hidden1_neuron_vector : temp_Prep_DE_hidden1_neuron_vector[",i,"] = ", temp_Prep_DE_hidden1_neuron_vector[i])
    #print("\n")
    
    return temp_Prep_DE_hidden1_neuron_vector

def Summing_two_hidden_size_lists_function(gamma_h_hidden, SUM_neg_DE_hidden_neuron_vector):
    
    # About: This function is to sum each nested list elements of two different lists
    # For, h(epoch) = -delta_E + gamma*h(epoch-1) or W = W + delta_W
    temp_final_hidden = []
    final_hidden = [None for i in range(len(gamma_h_hidden))]
    for i in range(len(gamma_h_hidden)):
        temp_hidden2_orig = deepcopy(gamma_h_hidden[i])
        temp_hidden2_delta = deepcopy(SUM_neg_DE_hidden_neuron_vector[i])
        for (item1, item2) in zip(temp_hidden2_orig, temp_hidden2_delta):
            temp_final_hidden.extend([item1 + item2])
        final_hidden[i] = deepcopy(temp_final_hidden)
        temp_final_hidden.clear()
    #print("Summing_two_hidden_size_lists_function : final_hidden = ", final_hidden)
            
    return final_hidden

# State the derivate of a function for each neuron weight/bias
def delta_E_function(W_new_out, W_new_HLN3, W_new_HLN2, W_new_HLN1, calculated_outputs, Activated_HLN3_list_for_DE, Activated_HLN2_list_for_DE, Activated_HLN1_list_for_DE, data_input_Vgs, data_input_Vds, data_outputs):
        
    # print("----------------------------------------------")
    # print("delta_E_function : W_new_out = ", W_new_out)
    # print("delta_E_function : W_new_HLN3 = ", W_new_HLN3)
    # print("delta_E_function : W_new_HLN3 = ", W_new_HLN3)
    # print("delta_E_function : W_new_HLN1 = ", W_new_HLN1)
    # print("\n")
    # print("delta_E_function : calculated_outputs = ", calculated_outputs)
    # for i in range(len(Activated_HLN3_list_for_DE)):
    #     print("delta_E_function : Activated_HLN3_list_for_DE(",i,") = ", Activated_HLN3_list_for_DE[i])
    # print("\n")
    
    # for i in range(len(Activated_HLN2_list_for_DE)):
    #     print("delta_E_function : Activated_HLN2_list_for_DE(",i,") = ", Activated_HLN2_list_for_DE[i])
    # print("\n")
    
    # for i in range(len(Activated_HLN1_list_for_DE)):
    #     print("delta_E_function : Activated_HLN1_list_for_DE(",i,") = ", Activated_HLN1_list_for_DE[i])
    # print("\n")
    
    # print("delta_E_function : calculated_outputs = ", calculated_outputs)
    # print("delta_E_function : Activated_HLN1_list_for_DE = ", Activated_HLN1_list_for_DE)
    # print("\n")
    # print("delta_E_function : Activated_HLN2_list_for_DE = ", Activated_HLN2_list_for_DE)
    # print("\n")
    
    Diff_Calc_Measure = []
    for i in range(len(data_outputs)):
        Diff_Calc_Measure.extend([calculated_outputs[i] - data_outputs[i]])

    
    # -------------------------------- DE Output neuron below ----------------------------------------
    # Prev_DE... vector is the collection of the derivatives of the weights/bias
           
    DE_Output_neuron_vector = DE_out_and_neg_DE_out_vector(Activated_HLN3_list_for_DE, Diff_Calc_Measure)
    #for i in range(len(DE_Output_neuron_vector)):
    #    print("delta_E_function : DE_Output_neuron_vector[",i,"] = ", DE_Output_neuron_vector[i])
    #print("\n")
    
    SUM_DE_Output_neuron_vector = [sum(i) for i in zip(*DE_Output_neuron_vector)]
    SUM_neg_DE_Output_neuron_vector = [-x for x in deepcopy(SUM_DE_Output_neuron_vector)]
    #print("delta_E_function : SUM_DE_Output_neuron_vector = ", SUM_DE_Output_neuron_vector)
    #print("delta_E_function : SUM_neg_DE_Output_neuron_vector = ", SUM_neg_DE_Output_neuron_vector)
    #print("\n")
    
    # -------------------------------- Hidden 3 neurons below ----------------------------------------
    # Delta E vector for the weights and the bias of the output neuron with entire data set(12 in total)
    # Below, Prep_#### with the multiplication of (calc-measure), contains all 12 samples
    
    SUM_DE_hidden3_neuron_vector = SUM_DE_hidden3_neuron_vector_func(Activated_HLN3_list_for_DE, Activated_HLN2_list_for_DE, Diff_Calc_Measure)
    #print("delta_E_function : SUM_DE_hidden3_neuron_vector = ", SUM_DE_hidden3_neuron_vector)
    #print("\n")   
    
    # Negate all the values
    SUM_neg_DE_hidden3_neuron_vector = [[-x for x in neuron] for neuron in deepcopy(SUM_DE_hidden3_neuron_vector)]
    #print("delta_E_function : SUM_neg_DE_hidden3_neuron_vector = ", SUM_neg_DE_hidden3_neuron_vector)
    #print("\n")   
    
    # -------------------------------- Hidden 2 neurons below ----------------------------------------
    # Delta E vector for the weights and the bias of the output neuron with entire data set(12 in total)
    # Below, Prep_#### with the multiplication of (calc-measure), contains all 12 samples
    
    Prep_DE_hidden2_neuron_vector = Func_Prep_DE_hidden2_neuron_vector(Activated_HLN3_list_for_DE, Activated_HLN2_list_for_DE, Activated_HLN1_list_for_DE, Diff_Calc_Measure)
    #for i in range(len(data_input_Vds)):
    #    print("delta_E_function : Prep_DE_hidden2_neuron_vector [",i,"] = ", Prep_DE_hidden2_neuron_vector[i])
    #print("\n")
    
    SUM_DE_hidden2_neuron_vector = deepcopy(Prep_DE_hidden2_neuron_vector[-1])
    for i in range(len(Prep_DE_hidden2_neuron_vector)-1):
        SUM_DE_hidden2_neuron_vector = np.array(Prep_DE_hidden2_neuron_vector[i]) + np.array(SUM_DE_hidden2_neuron_vector)
    #print("delta_E_function : SUM_DE_hidden2_neuron_vector = ", SUM_DE_hidden2_neuron_vector)
    #print("\n")
    
    # Negate all the values
    SUM_neg_DE_hidden2_neuron_vector = [[-x for x in neuron] for neuron in deepcopy(SUM_DE_hidden2_neuron_vector)]
    #print("delta_E_function : SUM_neg_DE_hidden2_neuron_vector = ", np.array(SUM_neg_DE_hidden2_neuron_vector))
    #print("\n")      
            
    # -------------------------------- Hidden 1 neurons below ----------------------------------------
    # Delta E vector for the weights and the bias of the output neuron with entire data set(12 in total)
    # Below, Prep_#### with the multiplication of (calc-measure), contains all 12 samples
    
    Prep_DE_hidden1_neuron_vector = Func_Prep_DE_hidden1_neuron_vector(Activated_HLN3_list_for_DE, Activated_HLN2_list_for_DE, Activated_HLN1_list_for_DE, Diff_Calc_Measure, data_input_Vgs, data_input_Vds)
    #for i in range(len(data_input_Vds)):
    #    print("delta_E_function : Prep_DE_hidden1_neuron_vector [",i,"] = ", Prep_DE_hidden1_neuron_vector[i])
    #print("\n")
    
    SUM_DE_hidden1_neuron_vector = deepcopy(Prep_DE_hidden1_neuron_vector[-1])
    for i in range(len(Prep_DE_hidden1_neuron_vector)-1):
        SUM_DE_hidden1_neuron_vector = np.array(Prep_DE_hidden1_neuron_vector[i]) + np.array(SUM_DE_hidden1_neuron_vector)
    #print("delta_E_function : SUM_DE_hidden1_neuron_vector = ", SUM_DE_hidden1_neuron_vector)
    #print("\n")
    
    # Negate all the values
    SUM_neg_DE_hidden1_neuron_vector = [[-x for x in neuron] for neuron in deepcopy(SUM_DE_hidden1_neuron_vector)]
    #print("delta_E_function : SUM_neg_DE_hidden1_neuron_vector = ", SUM_neg_DE_hidden1_neuron_vector)
    #print("\n")      
    
    
    return SUM_DE_Output_neuron_vector, SUM_neg_DE_Output_neuron_vector, SUM_DE_hidden3_neuron_vector, SUM_neg_DE_hidden3_neuron_vector, SUM_DE_hidden2_neuron_vector, SUM_neg_DE_hidden2_neuron_vector, SUM_DE_hidden1_neuron_vector, SUM_neg_DE_hidden1_neuron_vector

def gamma_function(Delta_E_W_out, Delta_E_W_hidden2, Delta_E_W_hidden1):
    
    #print("--------------------------------------------------------------------")
    # The length of DE_out and DE_hidden is 2
    # The first element of both is the present epoch and second is the previous epoch
    # print("gamma_function : Delta_E_W_out = ", Delta_E_W_out)
    # print("gamma_function : Delta_E_W_hidden2 = ", Delta_E_W_hidden2)
    # print("gamma_function : Delta_E_W_hidden1 = ", Delta_E_W_hidden1)
    # print("\n")
    
    num = 0         # initialize the numerator and the denominator to zero
    denom = 0
    
    for i in range(len(Delta_E_W_hidden2)):
        if (i == 0):
            for neuron in Delta_E_W_out[i]:
                num = num + pow(neuron,2)
                
            for neuron in Delta_E_W_hidden2[i]:
                for j in range(len(neuron)):
                    num = num + pow(neuron[j],2)
                    
            for neuron in Delta_E_W_hidden1[i]:
                for j in range(len(neuron)):
                    num = num + pow(neuron[j],2)
                    
        else:
            for neuron in Delta_E_W_out[i]:
                denom = denom + pow(neuron,2)
                
            for neuron in Delta_E_W_hidden2[i]:
                for j in range(len(neuron)):
                    denom = denom + pow(neuron[j],2)
            
            for neuron in Delta_E_W_hidden1[i]:
                for j in range(len(neuron)):
                    denom = denom + pow(neuron[j],2)

    num = sqrt(num)
    denom = sqrt(denom)
    gamma = num/denom
    
    return gamma

def line_search_function_h_hidden(W_new_out, W_new_HLN3, W_new_HLN2, W_new_HLN1, h_out, h_hidden3, h_hidden2, h_hidden1, data_input_Vgs, data_input_Vds, data_outputs):
    # check the line search method from prof's textbook (pg.60~)
    # etah_1 = min(=0) , etah_2 = max(= fixed number, user given)4

    count = 0
    #my decision for abs(etah2 - etah1)
    zero_threshold = 0.00000001
    #print("line_search_function_h_hidden: W_new_out = ", W_new_out)
    #print("line_search_function_h_hidden: W_new_HLN2 = ", W_new_HLN2)
    #print("line_search_function_h_hidden: W_new_HLN1 = ", W_new_HLN1)
    #print("line_search_function_h_hidden: h_out = ", h_out)
    #print("line_search_function_h_hidden: h_hidden2 = ", h_hidden2)
    #print("line_search_function_h_hidden: h_hidden1 = ", h_hidden1)
    #print("\n")
    
    #step 1
    etah1 = 0
    etah2 = 0.5
    etah3 = etah2 - 0.618*(etah2-etah1)
    etah4 = etah1 + 0.618*(etah2-etah1)
    W_out_original = W_new_out
    W_hidden3_original = W_new_HLN3
    W_hidden2_original = W_new_HLN2
    W_hidden1_original = W_new_HLN1
    h_out_original = h_out
    h_hidden3_original = h_hidden3
    h_hidden2_original = h_hidden2
    h_hidden1_original = h_hidden1
    #print("line_search_function_h_hidden: W_out_original : ", W_out_original)
    #print("line_search_function_h_hidden: W_hidden2_original : ", W_hidden2_original)
    #print("line_search_function_h_hidden: W_hidden1_original : ", W_hidden1_original)
    #print("\n")
    #print("line_search_function_h_hidden: h_out_original : ", h_out_original)
    #print("line_search_function_h_hidden: h_hidden2_original : ", h_hidden2_original)
    #print("line_search_function_h_hidden: h_hidden1_original : ", h_hidden1_original)
    #print("\n")
    
    #step2
    for i in range(1000):
    
        phi3 = function_for_phi_calculation(W_out_original, W_hidden3_original, W_hidden2_original, W_hidden1_original, h_out_original, h_hidden3_original, h_hidden2_original, h_hidden1_original, data_input_Vgs, data_input_Vds, data_outputs, etah3)
        #print("line_search_function_h_hidden: phi3 : ", phi3)
        
        phi4 = function_for_phi_calculation(W_out_original, W_hidden3_original, W_hidden2_original, W_hidden1_original, h_out_original, h_hidden3_original, h_hidden2_original, h_hidden1_original, data_input_Vgs, data_input_Vds, data_outputs, etah4)
        #print("line_search_function_h_hidden: phi4 : ", phi4)
        #print("\n")
        
        #------------------------------------------------------------------#
        if (phi3 > phi4):
            etah1 = etah3
            etah3 = etah4
            etah4 = etah1 + 0.618*(etah2-etah1)
            phi4 = function_for_phi_calculation(W_out_original, W_hidden3_original, W_hidden2_original, W_hidden1_original, h_out_original, h_hidden3_original, h_hidden2_original, h_hidden1_original, data_input_Vgs, data_input_Vds, data_outputs, etah4)
            
        else :
            etah2 = etah4
            etah4 = etah3
            etah3 = etah2 - 0.618*(etah2-etah1)
            phi3 = function_for_phi_calculation(W_out_original, W_hidden3_original, W_hidden2_original, W_hidden1_original, h_out_original, h_hidden3_original, h_hidden2_original, h_hidden1_original, data_input_Vgs, data_input_Vds, data_outputs, etah3)
            
        if (abs(etah2 - etah1) < zero_threshold):
            etah = etah3
            break
        count = count + 1
              
    print("line_search_function_h_hidden : count = ", count)
    print("line_search_function_h_hidden : etah = ", etah)
    return etah

def conjugate_gradient(user_Validation_error, W_out, W_HLN3, W_HLN2, W_HLN1, data_Tr_input_Vgs, data_Tr_input_Vds, data_Tr_output_Id, data_V_input_Vgs, data_V_input_Vds, data_V_output_Id, data_Te_input_Vgs, data_Te_input_Vds, data_Te_output_Id, max_epoch):
    
    Training_error_collection = []
    num_epoch_iterated = 0
    W_new_out = [W_out, None]
    W_new_HLN3 = [W_HLN3, None]
    W_new_HLN2 = [W_HLN2, None]
    W_new_HLN1 = [W_HLN1, None]
    h_out = [None, None]
    h_hidden3 = [None, None]
    h_hidden2 = [None, None]
    h_hidden1 = [None, None]
    Delta_E_W_out  = [None, None]
    Delta_E_W_hidden3  = [None, None]
    Delta_E_W_hidden2  = [None, None]
    Delta_E_W_hidden1  = [None, None]
    
    
    for epoch in range(max_epoch):
        print("----------------------------------------------")
        print("conjugate_gradient : epoch = ", epoch)
        calculated_outputs_Tr, epoch_HL3_activated_list, epoch_HL2_activated_list,epoch_HL1_activated_list = forward_propagate_3HL_with_Activated(W_new_out[0], W_new_HLN3[0], W_new_HLN2[0], W_new_HLN1[0], data_Tr_input_Vgs, data_Tr_input_Vds)
        FF_error_Tr = error_checking(calculated_outputs_Tr, data_Tr_output_Id)
        Training_error_collection.extend([FF_error_Tr])
        print("conjugate_gradient: calculated_outputs_Tr = ", calculated_outputs_Tr)
        print("conjugate_gradient: FF_error_Tr = ", FF_error_Tr)
        print("\n")
        
        calculated_outputs_V = forward_propagate_3HL(W_new_out[0], W_new_HLN3[0], W_new_HLN2[0], W_new_HLN1[0], data_V_input_Vgs, data_V_input_Vds)
        FF_error_V = error_checking(calculated_outputs_V, data_V_output_Id)
        print("conjugate_gradient: calculated_outputs_V = ", calculated_outputs_V)
        print("conjugate_gradient: FF_error_V = ", FF_error_V)
        print("\n")
        
        if ((FF_error_Tr < user_Validation_error) or (FF_error_V == FF_error_Tr)):
            break
        # print("conjugate_gradient: W_new_out[0] initial = ", W_new_out[0])
        # print("conjugate_gradient: W_new_HLN3[0] initial = ", W_new_HLN3[0])
        # print("conjugate_gradient: W_new_HLN2[0] initial = ", W_new_HLN2[0])
        # print("conjugate_gradient: W_new_HLN1[0] initial = ", W_new_HLN1[0])
        # print("\n")
        W_new_out_epoch0 = deepcopy(W_new_out[0])
        W_new_HLN3_epoch0 = deepcopy(W_new_HLN3[0])
        W_new_HLN2_epoch0 = deepcopy(W_new_HLN2[0])
        W_new_HLN1_epoch0 = deepcopy(W_new_HLN1[0])
        
        SUM_DE_Output_neuron_vector, SUM_neg_DE_Output_neuron_vector, SUM_DE_hidden3_neuron_vector, SUM_neg_DE_hidden3_neuron_vector, SUM_DE_hidden2_neuron_vector, SUM_neg_DE_hidden2_neuron_vector, SUM_DE_hidden1_neuron_vector, SUM_neg_DE_hidden1_neuron_vector = delta_E_function(W_new_out[0], W_new_HLN3[0], W_new_HLN2[0], W_new_HLN1[0], calculated_outputs_Tr, epoch_HL3_activated_list, epoch_HL2_activated_list, epoch_HL1_activated_list, data_Tr_input_Vgs, data_Tr_input_Vds, data_Tr_output_Id)
        Delta_E_W_out.insert(0, deepcopy(SUM_DE_Output_neuron_vector))
        Delta_E_W_hidden3.insert(0, deepcopy(SUM_DE_hidden3_neuron_vector))
        Delta_E_W_hidden2.insert(0, deepcopy(SUM_DE_hidden2_neuron_vector))
        Delta_E_W_hidden1.insert(0, deepcopy(SUM_DE_hidden1_neuron_vector))
        Delta_E_W_out.pop(-1)
        Delta_E_W_hidden3.pop(-1)
        Delta_E_W_hidden2.pop(-1)
        Delta_E_W_hidden1.pop(-1)
        
        print("conjugate_gradient: SUM_DE_Output_neuron_vector = ", SUM_DE_Output_neuron_vector)
        print("conjugate_gradient: SUM_DE_hidden3_neuron_vector = ", SUM_DE_hidden3_neuron_vector)
        print("conjugate_gradient: SUM_DE_hidden2_neuron_vector = ", SUM_DE_hidden2_neuron_vector)
        print("conjugate_gradient: SUM_DE_hidden1_neuron_vector = ", SUM_DE_hidden1_neuron_vector)
        print("\n")
        print("conjugate_gradient: SUM_neg_DE_Output_neuron_vector = ", SUM_neg_DE_Output_neuron_vector)
        print("conjugate_gradient: SUM_neg_DE_hidden3_neuron_vector = ", np.array(SUM_neg_DE_hidden3_neuron_vector))
        print("conjugate_gradient: SUM_neg_DE_hidden2_neuron_vector = ", np.array(SUM_neg_DE_hidden2_neuron_vector))
        print("conjugate_gradient: SUM_neg_DE_hidden1_neuron_vector = ", np.array(SUM_neg_DE_hidden1_neuron_vector))
        print("\n")
        print("conjugate_gradient: Delta_E_W_out = ", Delta_E_W_out)
        print("conjugate_gradient: Delta_E_W_hidden3 = ", Delta_E_W_hidden3)
        print("conjugate_gradient: Delta_E_W_hidden2 = ", Delta_E_W_hidden2)
        print("conjugate_gradient: Delta_E_W_hidden1 = ", Delta_E_W_hidden1)
        print("\n")

        if (epoch == 0):
            print("conjugate_gradient: epoch = ", epoch)
            h_out.insert(0, deepcopy(SUM_neg_DE_Output_neuron_vector))
            h_hidden3.insert(0, deepcopy(SUM_neg_DE_hidden3_neuron_vector))
            h_hidden2.insert(0, deepcopy(SUM_neg_DE_hidden2_neuron_vector))
            h_hidden1.insert(0, deepcopy(SUM_neg_DE_hidden1_neuron_vector))
            h_out.pop(-1)
            h_hidden3.pop(-1)
            h_hidden2.pop(-1)
            h_hidden1.pop(-1)
            
            h_hidden3_epoch0 = deepcopy(h_hidden3[0])
            h_hidden2_epoch0 = deepcopy(h_hidden2[0])
            h_hidden1_epoch0 = deepcopy(h_hidden1[0])
            h_out_epoch0 = deepcopy(h_out[0])
            
            etah = line_search_function_h_hidden(W_new_out_epoch0, W_new_HLN3_epoch0, W_new_HLN2_epoch0, W_new_HLN1_epoch0, h_out_epoch0, h_hidden3_epoch0, h_hidden2_epoch0, h_hidden1_epoch0, data_Tr_input_Vgs, data_Tr_input_Vds, data_Tr_output_Id)
            print("conjugate_gradient: etah (epoch =",epoch,") = ", etah)
            print("\n")
            
            W_out_final, W_hidden3_final, W_hidden2_final, W_hidden1_final = W_or_DE_Mul_with_etah(W_new_out_epoch0, W_new_HLN3_epoch0, W_new_HLN2_epoch0, W_new_HLN1_epoch0, h_out_epoch0, h_hidden3_epoch0, h_hidden2_epoch0, h_hidden1_epoch0, etah)
            # print("conjugate_gradient: W_out_final (epoch =",epoch,") = ", W_out_final)
            # print("conjugate_gradient: W_hidden2_final (epoch =",epoch,") = ", W_hidden2_final)
            # print("conjugate_gradient: W_hidden1_final (epoch =",epoch,") = ", W_hidden1_final)
            # print("\n")
            
            W_new_out.insert(0, deepcopy(W_out_final))
            W_new_HLN3.insert(0, deepcopy(W_hidden3_final))
            W_new_HLN2.insert(0, deepcopy(W_hidden2_final))
            W_new_HLN1.insert(0, deepcopy(W_hidden1_final))
            # print("conjugate_gradient: W_new_out (epoch =",epoch,") = ", W_new_out)
            # print("conjugate_gradient: W_new_HLN2 (epoch =",epoch,") = ", W_new_HLN2)
            # print("conjugate_gradient: W_new_HLN1 (epoch =",epoch,") = ", W_new_HLN1)
            # print("\n")
            
            W_new_out.pop(-1)
            W_new_HLN3.pop(-1)
            W_new_HLN2.pop(-1)
            W_new_HLN1.pop(-1)
            # print("conjugate_gradient: W_new_out (epoch =",epoch,") = ", W_new_out)
            # print("conjugate_gradient: W_new_HLN2 (epoch =",epoch,") = ", W_new_HLN2)
            # print("conjugate_gradient: W_new_HLN1 (epoch =",epoch,") = ", W_new_HLN1)
            # print("\n")
            
        else:
            # print("conjugate_gradient: SUM_DE_Output_neuron_vector (epoch >= ",epoch,") = ", SUM_DE_Output_neuron_vector)
            # print("conjugate_gradient: SUM_DE_hidden2_neuron_vector (epoch >= ",epoch,") = ", SUM_DE_hidden2_neuron_vector)
            # print("conjugate_gradient: SUM_DE_hidden1_neuron_vector (epoch >= ",epoch,") = ", SUM_DE_hidden1_neuron_vector)
            # print("\n")
            
            # print("conjugate_gradient: Delta_E_W_out (epoch >= ",epoch,") = ", Delta_E_W_out)
            # print("conjugate_gradient: Delta_E_W_hidden2 (epoch >= ",epoch,") = ", Delta_E_W_hidden2)
            # print("conjugate_gradient: Delta_E_W_hidden1 (epoch >= ",epoch,") = ", Delta_E_W_hidden1)
            # print("\n")
            
            gamma = gamma_function(Delta_E_W_out, Delta_E_W_hidden2, Delta_E_W_hidden1)
            # print("conjugate_gradient: gamma (epoch >= ",epoch,") = ", gamma)
            # print("\n")
            # print("conjugate_gradient: h_out (epoch >= ",epoch,") = ", h_out)
            # print("conjugate_gradient: h_hidden2 (epoch >= ",epoch,") = ", h_hidden2)
            # print("conjugate_gradient: h_hidden1 (epoch >= ",epoch,") = ", h_hidden1)
            # print("\n")
            # print("conjugate_gradient: SUM_neg_DE_Output_neuron_vector (epoch >= ",epoch,") = ", SUM_neg_DE_Output_neuron_vector)
            # print("conjugate_gradient: SUM_neg_DE_hidden2_neuron_vector (epoch >= ",epoch,") = ", SUM_neg_DE_hidden2_neuron_vector)
            # print("conjugate_gradient: SUM_neg_DE_hidden1_neuron_vector (epoch >= ",epoch,") = ", SUM_neg_DE_hidden1_neuron_vector)
            # print("\n")
            
            h_out.pop(-1)
            h_hidden3.pop(-1)
            h_hidden2.pop(-1)
            h_hidden1.pop(-1)
            print("conjugate_gradient: h_out (epoch >= ",epoch,") DEBUG = ", h_out)
            print("conjugate_gradient: h_hidden3 (epoch >= ",epoch,") DEBUG = ", h_hidden3)
            print("conjugate_gradient: h_hidden2 (epoch >= ",epoch,") DEBUG = ", h_hidden2)
            print("conjugate_gradient: h_hidden1 (epoch >= ",epoch,") DEBUG = ", h_hidden1)
            print("\n")
            
            # Computation below is for gamma * h(epoch - 1)
            gamma_h_out = [x*gamma for x in deepcopy(h_out[0])]
            gamma_h_hidden3 = [[x*gamma for x in neuron] for neuron in deepcopy(h_hidden3[0])]
            gamma_h_hidden2 = [[x*gamma for x in neuron] for neuron in deepcopy(h_hidden2[0])]
            gamma_h_hidden1 = [[x*gamma for x in neuron] for neuron in deepcopy(h_hidden1[0])]
            
            print("conjugate_gradient: SUM_neg_DE_Output_neuron_vector (epoch >= ",epoch,") DEBUG = ", SUM_neg_DE_Output_neuron_vector)
            print("conjugate_gradient: SUM_neg_DE_hidden3_neuron_vector (epoch >= ",epoch,") DEBUG = ", SUM_neg_DE_hidden3_neuron_vector)
            print("conjugate_gradient: SUM_neg_DE_hidden2_neuron_vector (epoch >= ",epoch,") DEBUG = ", SUM_neg_DE_hidden2_neuron_vector)
            print("conjugate_gradient: SUM_neg_DE_hidden1_neuron_vector (epoch >= ",epoch,") DEBUG = ", SUM_neg_DE_hidden1_neuron_vector)
            print("\n")
            print("conjugate_gradient: gamma_h_out (epoch >= ",epoch,") DEBUG = ", gamma_h_out)
            print("conjugate_gradient: gamma_h_hidden3 (epoch >= ",epoch,") DEBUG = ", gamma_h_hidden3)
            print("conjugate_gradient: gamma_h_hidden2 (epoch >= ",epoch,") DEBUG = ", gamma_h_hidden2)
            print("conjugate_gradient: gamma_h_hidden1 (epoch >= ",epoch,") DEBUG = ", gamma_h_hidden1)
            print("\n")
            
            # Sum the h vectors : h(epoch) = -delta_E + gamma*h(epoch-1)
            final_h_out = np.array(gamma_h_out) + np.array(SUM_neg_DE_Output_neuron_vector)
            final_hidden3 = np.array(SUM_neg_DE_hidden3_neuron_vector) + np.array(gamma_h_hidden3)
            final_hidden2 = np.array(SUM_neg_DE_hidden2_neuron_vector) + np.array(gamma_h_hidden2)
            final_hidden1 = np.array(SUM_neg_DE_hidden1_neuron_vector) + np.array(gamma_h_hidden1)
            
            print("conjugate_gradient: final_h_out (epoch >= ",epoch,") DEBUG = ", final_h_out)
            print("conjugate_gradient : final_hidden3 (epoch >= ",epoch,") DEBUG = ", final_hidden3)
            print("conjugate_gradient : final_hidden2 (epoch >= ",epoch,") DEBUG = ", final_hidden2)
            print("conjugate_gradient : final_hidden1 (epoch >= ",epoch,") DEBUG = ", final_hidden1)
            print("\n")
            
            h_out.insert(0,deepcopy(final_h_out))
            h_hidden3.insert(0,deepcopy(final_hidden3))
            h_hidden2.insert(0,deepcopy(final_hidden2))
            h_hidden1.insert(0,deepcopy(final_hidden1))
            # print("conjugate_gradient: h_out (epoch >= ",epoch,") = ", h_out)
            # print("conjugate_gradient: h_hidden2 (epoch >= ",epoch,") = ", h_hidden2)
            # print("conjugate_gradient: h_hidden1 (epoch >= ",epoch,") = ", h_hidden1)
            # print("\n")
            
            etah = line_search_function_h_hidden(W_new_out[0], W_new_HLN3[0], W_new_HLN2[0], W_new_HLN1[0], h_out[0], h_hidden3[0], h_hidden2[0], h_hidden1[0], data_Tr_input_Vgs, data_Tr_input_Vds, data_Tr_output_Id)
            W_out_final, W_hidden3_final, W_hidden2_final, W_hidden1_final = W_or_DE_Mul_with_etah(W_new_out[0], W_new_HLN3[0], W_new_HLN2[0], W_new_HLN1[0], h_out[0], h_hidden3[0], h_hidden2[0], h_hidden1[0], etah)
            # print("conjugate_gradient: etah (epoch >= ",epoch,") = ", etah)
            # print("conjugate_gradient: W_out_final (epoch >= ",epoch,") = ", W_out_final)
            # print("conjugate_gradient: W_hidden2_final (epoch >= ",epoch,") = ", W_hidden2_final)
            # print("conjugate_gradient: W_hidden1_final (epoch >= ",epoch,") = ", W_hidden1_final)
            # print("\n")
            
            W_new_out.insert(0, W_out_final)
            W_new_HLN3.insert(0, W_hidden3_final)
            W_new_HLN2.insert(0, W_hidden2_final)
            W_new_HLN1.insert(0, W_hidden1_final)
            # print("conjugate_gradient: W_new_out (epoch >= ",epoch,") = ", W_new_out)
            # print("conjugate_gradient: W_new_HLN2 (epoch >= ",epoch,") = ", W_new_HLN2)
            # print("conjugate_gradient: W_new_HLN1 (epoch >= ",epoch,") = ", W_new_HLN1)
            # print("\n")
            
            W_new_out.pop(-1)
            W_new_HLN3.pop(-1)
            W_new_HLN2.pop(-1)
            W_new_HLN1.pop(-1)
            # print("conjugate_gradient: W_new_out (epoch >= ",epoch,") = ", W_new_out)
            # print("conjugate_gradient: W_new_HLN2 (epoch >= ",epoch,") = ", W_new_HLN2)
            # print("conjugate_gradient: W_new_HLN1 (epoch >= ",epoch,") = ", W_new_HLN1)
            # print("\n")
            
            
        num_epoch_iterated = num_epoch_iterated + 1
        #print("conjugate_gradient: num_epoch_iterated (epoch >= ",epoch,") = ", num_epoch_iterated)    
    
    calculated_outputs_V = forward_propagate_3HL(W_new_out[0], W_new_HLN3[0], W_new_HLN2[0], W_new_HLN1[0], data_V_input_Vgs, data_V_input_Vds)
    Test_error_collection = error_checking_Test_set(calculated_outputs_V, data_Te_output_Id)
        
    return W_new_out[0], W_new_HLN3[0], W_new_HLN2[0], W_new_HLN1[0], num_epoch_iterated, Training_error_collection, Test_error_collection

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
print("Tr_Id_min : " , Tr_Id_min)
print("V_Id_min : " , V_Id_min)
print("Te_Id_min : " , Te_Id_min)
print("\n")

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
        

temp_input = 2
# Initialize the weight values when a number was typed by an user in GUI
hidden1 = 5
if (hidden1 > 0):
    temp_W_HLN1 = [[2*random()-1 for i in range(temp_input + 1)] for i in range(hidden1)] # Initializing random weights and biases in a range of -1 to 1
hidden2 = 5
if (hidden2 > 0):
    temp_W_HLN2 = [[2*random()-1 for i in range(hidden2 + 1)] for i in range(hidden2)] # Initializing random weights and biases in a range of -1 to 1
hidden3 = 5
if (hidden2 > 0):              
    temp_W_HLN3 = [[2*random()-1 for i in range(hidden3 + 1)] for i in range(hidden3)] # Initializing random weights and biases in a range of -1 to 1

W_out = [2*random()-1 for i in range(len(temp_W_HLN3) + 1) ]
W_HLN3 = deepcopy(temp_W_HLN3)
W_HLN2 = deepcopy(temp_W_HLN2)
W_HLN1 = deepcopy(temp_W_HLN1)

# Hidden neurons will be activated with the sigmoid function
# Output neuron will be activated with the linear function
print("W_out : " , W_out)
print("\n")
print("W_HLN3 : " , W_HLN3)
print("\n")
print("W_HLN2 : " , W_HLN2)
print("\n")
print("W_HLN1 : " , W_HLN1)
print("\n")

max_epoch = 50;
user_Validation_error = 0.1;
ALI_HIDDEN_VALUE = 3
num_hidden_layer = ALI_HIDDEN_VALUE

if (num_hidden_layer == 1):
    [W_new_out, W_new_HLN1, num_epoch_iter, Training_error_collection, Test_error_collection] = conjugate_gradient(user_Validation_error, W_out, W_HLN1, data_Tr_input_Vgs, data_Tr_input_Vds, scaled_Tr_output_Id, data_V_input_Vgs, data_V_input_Vds, scaled_V_output_Id, data_Te_input_Vgs, data_Te_input_Vds, scaled_Te_output_Id, max_epoch)
elif (num_hidden_layer == 2):
    [W_new_out, W_new_HLN2, W_new_HLN1, num_epoch_iter, Training_error_collection, Test_error_collection] = conjugate_gradient(user_Validation_error, W_out, W_HLN2, W_HLN1, data_Tr_input_Vgs, data_Tr_input_Vds, scaled_Tr_output_Id, data_V_input_Vgs, data_V_input_Vds, scaled_V_output_Id, data_Te_input_Vgs, data_Te_input_Vds, scaled_Te_output_Id, max_epoch)
else :
    [W_new_out, W_new_HLN3, W_new_HLN2, W_new_HLN1, num_epoch_iter, Training_error_collection, Test_error_collection] = conjugate_gradient(user_Validation_error, W_out, W_HLN3, W_HLN2, W_HLN1, data_Tr_input_Vgs, data_Tr_input_Vds, scaled_Tr_output_Id, data_V_input_Vgs, data_V_input_Vds, scaled_V_output_Id, data_Te_input_Vgs, data_Te_input_Vds, scaled_Te_output_Id, max_epoch)

print("W_new_out : " , W_new_out)
print("W_new_HLN3 : " , W_new_HLN3)
print("W_new_HLN2 : " , W_new_HLN2)
print("W_new_HLN1 : " , W_new_HLN1)
print("num_epoch_iter : " , num_epoch_iter)
print("\n")
print("Training_error_collection : " , Training_error_collection)
print("Test_error_collection : " , Test_error_collection)
print("\n")    

#-----------------------NOTE----------------------------
#The structure of the ANN with one hidden layer
#                    Id
#        # of Hidden Neurons here
#                Vgs     Vds