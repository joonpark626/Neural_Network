from copy import deepcopy
from wsgiref.simple_server import demo_app
import numpy as np
from math import *
import pandas as pd
import csv
from random import random

#Transfer neuron activation sigmoid function
def transfer_sigmoid(activation):
    return 1.0/(1.0 + exp(-activation))

def forward_propagate(W_new_HLN1, W_new_out, data_input_Vgs, data_input_Vds):
    calculated_outputs = []
    HL1_activated_list = []
    HL1_to_Output_Neuron_list = []
    temp = 0
    
    #print("---------------------------------------------------------------")
    #print("forward_propagate: W_new_HLN1 = ", W_new_HLN1)
    #print("forward_propagate: W_new_out = ", W_new_out)
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

        for j in range(len(W_new_out)):
            if (j != (len(W_new_out) - 1)):
                temp = temp + W_new_out[j]*HL1_activated_list[j]
            else:
                temp = temp + W_new_out[j]
        HL1_to_Output_Neuron_list.extend([transfer_sigmoid(temp)])
        temp = 0
        
        calculated_outputs.extend(HL1_to_Output_Neuron_list.copy())
        HL1_activated_list.clear()
        HL1_to_Output_Neuron_list.clear()
    
    #for i in range(len(calculated_outputs)):
    #    calculated_outputs[i] = round(calculated_outputs[i], 4)
    
    #print("forward_propagate: calculated_outputs = ", calculated_outputs)    
    return calculated_outputs

def forward_propagate_newWeights(W_new_HLN1, W_new_out, data_input_Vgs, data_input_Vds):
    
    calculated_outputs = []
    HL1_activated_list = []
    HL1_to_Output_Neuron_list = []
    HLN1_list_for_DE = []
    temp = 0
    
    #print("---------------------------------------------------------------")
    #print("forward_propagate: W_new_HLN1 = ", W_new_HLN1)
    #print("forward_propagate: W_new_out = ", W_new_out)
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
            
        HLN1_list_for_DE.append(HL1_activated_list.copy())
        
        for j in range(len(W_new_out)):
            if (j != (len(W_new_out) - 1)):
                temp = temp + W_new_out[j]*HL1_activated_list[j]
            else:
                temp = temp + W_new_out[j]
        HL1_to_Output_Neuron_list.extend([transfer_sigmoid(temp)])
        temp = 0
        
        calculated_outputs.extend(HL1_to_Output_Neuron_list.copy())
        HL1_activated_list.clear()
        HL1_to_Output_Neuron_list.clear()
        
    #print("forward_propagate_newWeights : calculated_outputs FINAL = ", calculated_outputs)    
    #print("\n")
    return calculated_outputs, HLN1_list_for_DE

# Comparing the calculated and measured outputs for initial Feedforward
def error_checking_for_etah(W_new, inputs, data_outputs):
    
    Calculated_Outputs = forward_propagate(W_new, inputs)
    
    error = 0
    for i in range(len(Calculated_Outputs)):
        error += 0.5*(pow(Calculated_Outputs[i] - data_outputs[i], 2))
    
    return error

# Comparing the calculated and measured outputs for conjugate gradient iteration
def error_checking(W_new_HLN1, W_new_out, data_input_Vgs, data_input_Vds, data_outputs):
    #print("---------------------------------------------------------------")
    # print("error_checking : W_new_HLN1 = ", W_new_HLN1)
    # print("error_checking : W_new_out = ", W_new_out)
    # print("error_checking : data_input_Vds = ", data_input_Vds)
    # print("error_checking : data_input_Vgs = ", data_input_Vgs)
    calculated_outputs = forward_propagate(W_new_HLN1, W_new_out, data_input_Vds, data_input_Vgs)
    error = 0
    for i in range(len(calculated_outputs)):
        error += 0.5*(pow(calculated_outputs[i] - data_outputs[i], 2))
    #print("error_checking : error = ", error)
    return error

def DE_out_vector_per_sample(Activated_HLN1_list_for_DE, data_input_Vds, data_input_Vgs):
    temp = []
    DE_OutNeuron_vector = []
    for i in range(len(Activated_HLN1_list_for_DE)):
        temp.extend([pow(Activated_HLN1_list_for_DE[i],2)*(1 - Activated_HLN1_list_for_DE[i])*data_input_Vgs])
        temp.extend([pow(Activated_HLN1_list_for_DE[i],2)*(1 - Activated_HLN1_list_for_DE[i])*data_input_Vds])
        temp.extend([pow(Activated_HLN1_list_for_DE[i],2)*(1 - Activated_HLN1_list_for_DE[i])])
        DE_OutNeuron_vector.append(temp.copy())
        temp.clear()
    return DE_OutNeuron_vector

def W_hidden_vector_comb(W_hidden_original, h_hidden_original, etah):

    #print("line_search_function_h_hidden_vector_comb : h_hidden_original = ", h_hidden_original)
    #print("line_search_function_h_hidden_vector_comb : W_hidden_original = ", W_hidden_original)
    #print("\n") 
    
    W_hidden_delta = []
    temp_hidden = []
    for neuron in h_hidden_original:
        for i in range(len(neuron)):
            temp_hidden.extend([neuron[i]*etah])
        W_hidden_delta.append(deepcopy(temp_hidden))
        temp_hidden.clear()
    #print("line_search_function_h_hidden: W_hidden_delta DEBUG1 : ", W_hidden_delta)
    #print("\n")
    
    temp = []
    temp_W_for_phi = []
    #for neuron in range(len(temp_delta)):
    for i in range(len(W_hidden_delta)):
            temp.append(W_hidden_delta[i])
            temp.append(W_hidden_original[i])
            temp_W_for_phi.append(deepcopy(temp))
            temp.clear()

    #for i in range(len(temp_W_for_phi)):
    #    print("line_search_function_h_hidden_vector_comb : temp_W_for_phi[",i,"] = ", temp_W_for_phi[i])
    #print("\n")        
    
    W_for_phi = []
    temp_temp = [0, 0, 0]
    
    for layer in temp_W_for_phi:
        for sample in layer:
            for i in range(len(sample)):
                temp_temp[i] = temp_temp[i] + sample[i]
        W_for_phi.append(deepcopy(temp_temp))
        temp_temp.clear()
        temp_temp.extend([0, 0, 0])
    #print("line_search_function_h_hidden_vector_comb : W_for_phi = ", W_for_phi)
    #print("\n")
    
    return W_for_phi

def function_for_final_weight(W_hidden_original, W_out_original, h_hidden_original, h_out_original,  data_input_Vgs, data_input_Vds, data_outputs, etah):
    
    W_hidden_for_phi = []
    W_out_for_phi = []
    W_out_delta = []
    
    #print("line_search_function_h_hidden_vector_comb : h_hidden_original Problem = ", h_hidden_original)
    #print("line_search_function_h_hidden_vector_comb : h_out_original = ", h_out_original)
    
    W_hidden_for_phi = W_hidden_vector_comb(W_hidden_original, h_hidden_original, etah)
    #print("line_search_function_h_hidden: W_hidden_for_phi : ", W_hidden_for_phi)
        
    for i in range(len(h_out_original)):
        W_out_delta.extend([h_out_original[i]*etah])
    #print("line_search_function_h_hidden: W_out_delta DEBUG1 : ", W_out_delta)
        
    for (orig, delta) in zip(W_out_original, W_out_delta):
        W_out_for_phi.extend([orig + delta])
    #print("line_search_function_h_hidden: W_out_for_phi : ", W_out_for_phi)
    
    W_hidden_final = deepcopy(W_hidden_for_phi)
    W_out_final = deepcopy(W_out_for_phi)
    
    #print("function_for_final_weight: W_hidden_final : ", W_hidden_final)
    #print("function_for_final_weight: W_out_final : ", W_out_final)
    #print("\n")
    
    return W_out_final, W_hidden_final

def function_for_phi_calculation(W_hidden_original, W_out_original, h_hidden_original, h_out_original,  data_input_Vgs, data_input_Vds, data_outputs, etah):
    
    W_hidden_for_phi = []
    W_out_for_phi = []
    W_hidden_delta = []
    W_out_delta = []

    W_hidden_for_phi = W_hidden_vector_comb(W_hidden_original, W_hidden_original, etah)
    #print("line_search_function_h_hidden: W_hidden_for_phi : ", W_hidden_for_phi)
        
    for i in range(len(h_out_original)):
        W_out_delta.extend([h_out_original[i]*etah])
    #print("line_search_function_h_hidden: W_out_delta DEBUG1 : ", W_out_delta)
        
    for (orig, delta) in zip(W_out_original, W_out_delta):
        W_out_for_phi.extend([orig + delta])
    #print("line_search_function_h_hidden: W_out_for_phi : ", W_out_for_phi)
    
    phi3 = error_checking(W_hidden_for_phi, W_out_for_phi, data_input_Vgs, data_input_Vds, data_outputs)
    
    return phi3

def SUM_DE_hidden_neuron_vector_function(DE_hidden_neuron_vector):
    
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #                   Hidden neurons organization
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    
    # Used for both SUM_DE_hidden_neuron_vector and SUM_neg_DE_hidden_neuron_vector
    temp_hidden = [None for i in range(len(DE_hidden_neuron_vector[0]))]
    temp1 = []
    for neuron in range(len(temp_hidden)):                      # Inside the each element, neuron will have all the sample values 
        for i in range(len(DE_hidden_neuron_vector)):           # Total number of samples = 12
            for layer in DE_hidden_neuron_vector:               # the current neuron values per each sample
                for k in range(len(layer)):                     # Accessing one neuron within the hidden layer
                    if (k == neuron):
                        temp1.append(deepcopy(layer[neuron]))
            temp_hidden[neuron] = deepcopy(temp1)
            temp1.clear()

    #for i in range(len(temp_hidden)):
    #    print("delta_E_function : temp_hidden [",i,"] = ", temp_hidden[i])
    #print("\n")
    
    SUM_DE_hidden_neuron_vector = []
    temp_temp = [0, 0, 0]
    for layer in temp_hidden:
        #print("delta_E_function : layer = ", layer)
        #print("\n")
        for sample in layer:
            #print("delta_E_function : sample = ", sample)
            for j in range(len(sample)):
                temp_temp[j] = temp_temp[j] + sample[j]
                #print("delta_E_function : temp_temp[",j,"] = ", temp_temp[j])
        SUM_DE_hidden_neuron_vector.append(deepcopy(temp_temp))
        temp_temp.clear()
        temp_temp.extend([0, 0, 0])
    #print("delta_E_function : SUM_DE_hidden_neuron_vector = ", SUM_DE_hidden_neuron_vector)
    #print("\n")
    
    return SUM_DE_hidden_neuron_vector

def function_gamma_x_prev_h_epoch(h_out, h_hidden, gamma):
    
    #print("function_gamma_x_prev_h_epoch : h_out = ", h_out)
    #print("function_gamma_x_prev_h_epoch : h_hidden = ", h_hidden)
    #print("function_gamma_x_prev_h_epoch : gamma = ", gamma)
    #print("\n")
    
    delta_h_out = []
    delta_h_hidden = []
    temp = []
    
    for weight in h_out:
        delta_h_out.extend([weight*gamma])
    #print("function_gamma_x_prev_h_epoch : delta_h_out = ", delta_h_out)
    
    for neuron in h_hidden:
        for i in range(len(neuron)):
            temp.extend([neuron[i]*gamma])
        delta_h_hidden.append(deepcopy(temp))
        temp.clear()
    #print("function_gamma_x_prev_h_epoch : delta_h_hidden = ", delta_h_hidden)
    #print("\n")
    
    return delta_h_out, delta_h_hidden


def function_sum_neg_deltaE_delta_h(delta_h_out, delta_h_hidden, neg_DE_Output_neuron_vector, neg_DE_hidden_neuron_vector):

    # print("function_sum_neg_deltaE_delta_h : delta_h_out = ", delta_h_out)
    # print("function_sum_neg_deltaE_delta_h : delta_h_hidden = ", delta_h_hidden)
    # print("function_sum_neg_deltaE_delta_h : neg_DE_Output_neuron_vector = ", neg_DE_Output_neuron_vector)
    # print("function_sum_neg_deltaE_delta_h : neg_DE_hidden_neuron_vector = ", neg_DE_hidden_neuron_vector)
    # print("\n")
    
    temp_h_out = []
    for i in range(len(delta_h_out)):
        temp_h_out.extend([delta_h_out[i] + neg_DE_Output_neuron_vector[i]])
    final_h_out = deepcopy(temp_h_out)
    #print("function_sum_neg_deltaE_delta_h : final_h_out = ", final_h_out)
    
    temp = []
    temp_h_hidden = []
    for i in range(len(delta_h_hidden)):
        temp.append(delta_h_hidden[i])
        temp.append(neg_DE_hidden_neuron_vector[i])
        temp_h_hidden.append(deepcopy(temp))
        temp.clear()
    
    final_h_hidden = []
    temp_temp = [0 for i in range(3)]
    for neuron in temp_h_hidden:
        for weight in neuron:
            for i in range(len(weight)):
                temp_temp[i] = temp_temp[i] + weight[i]
        final_h_hidden.append(deepcopy(temp_temp))
        temp_temp.clear()
        temp_temp.extend([0,0,0])
    #print("function_sum_neg_deltaE_delta_h : final_h_hidden = ", final_h_hidden)    
    #print("\n")
    
    return final_h_out, final_h_hidden
    
# State the derivate of a function for each neuron weight/bias
def delta_E_function(W_new_HLN1, W_new_out, data_input_Vgs, data_input_Vds, data_outputs):
        
    #print("----------------------------------------------")
    # print("delta_E_function : W_new_HLN1 DEBUG1 = ", W_new_HLN1)
    # print("delta_E_function : W_new_out DEBUG1 = ", W_new_out)
    
    calculated_outputs, Activated_HLN1_list_for_DE = forward_propagate_newWeights(W_new_HLN1, W_new_out, data_input_Vgs, data_input_Vds)
    # print("delta_E_function : calculated_outputs = ", calculated_outputs)
    
    Diff_Calc_Measure = []
    for i in range(len(data_input_Vds)):
        Diff_Calc_Measure.extend([calculated_outputs[i] - data_outputs[i]])
    # print("delta_E_function : Diff_Calc_Measure = ", Diff_Calc_Measure)
    # print("\n")
    
    # -------------------------------- Output neuron below ----------------------------------------
    # Prev_DE... vector is the collection of the derivatives of the weights/bias
    temp_Activated_HLN1_list_for_DE = deepcopy(Activated_HLN1_list_for_DE)
    
    Prev_DE_Output_neuron_vector = []
    for sample in temp_Activated_HLN1_list_for_DE:
        sample.extend([1])
        Prev_DE_Output_neuron_vector.append(sample)
    
    for element in Prev_DE_Output_neuron_vector:
        for i in range(len(element)):
            element[i] = element[i]*Diff_Calc_Measure[i] 
    DE_Output_neuron_vector = Prev_DE_Output_neuron_vector.copy()
    
    # for i in range(len(DE_Output_neuron_vector)):
    #     print("delta_E_function : DE_Output_neuron_vector [",i,"] = ", DE_Output_neuron_vector[i])
    # print("\n")
    
    neg_DE_Output_neuron_vector = deepcopy(DE_Output_neuron_vector)
    for sample in neg_DE_Output_neuron_vector:
        for i in range(len(sample)):
            sample[i] = sample[i]*-1
            
    # for i in range(len(neg_DE_Output_neuron_vector)):
    #     print("delta_E_function : neg_DE_Output_neuron_vector [",i,"] = ", neg_DE_Output_neuron_vector[i])
    # print("\n")   
    
    # -------------------------------- Hidden neurons below ----------------------------------------
    # Delta E vector for the weights and the bias of the output neuron with entire data set(12 in total)
    # Prev_#### means only the vector portion, not with the multiplication of (calc-measure)
    Prev_DE_hidden_neuron_vector = [None for i in range(len(data_input_Vds))]
    for i in range(len(data_input_Vds)):
        Prev_DE_hidden_neuron_vector[i] = DE_out_vector_per_sample(Activated_HLN1_list_for_DE[i], data_input_Vgs[i], data_input_Vds[i])
    # for i in range(len(data_input_Vds)):
    #     print("delta_E_function : Prev_DE_hidden_neuron_vector[",i,"] = ", Prev_DE_hidden_neuron_vector[i])
    # print("\n")
    
    for layer in Prev_DE_hidden_neuron_vector:
        for element in layer:
            for i in range(len(element)):
                element[i] = element[i]*Diff_Calc_Measure[i]
    DE_hidden_neuron_vector = Prev_DE_hidden_neuron_vector.copy()
    
    
    # Negating the delta E vector of hidden neurons
    temp_hidden_neurons = []
    temp_hidden_elements = []
    temp_hidden_weights = []
    temp_DE_hidden_neuron_vector = deepcopy(DE_hidden_neuron_vector)
    for layer in temp_DE_hidden_neuron_vector:
        for element in layer:
            for i in range(len(element)):
                element[i] = element[i]*-1
                temp_hidden_weights.extend([element[i]])
            temp_hidden_elements.append(deepcopy(temp_hidden_weights))
            temp_hidden_weights.clear()
        temp_hidden_neurons.append(deepcopy(temp_hidden_elements))
        temp_hidden_elements.clear()
    neg_DE_hidden_neuron_vector = deepcopy(temp_hidden_neurons)
            
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Output neuron organization
    
    # Summing all the delta E values (samples = epoch) to determine the overall delta E vector
    temp_out = [0 for i in range(len(DE_Output_neuron_vector[0]))]
    for sample in DE_Output_neuron_vector:
        for i in range(len(sample)):
            temp_out[i] = temp_out[i] + sample[i]
    SUM_DE_Output_neuron_vector = deepcopy(temp_out)
    #print("delta_E_function : SUM_DE_Output_neuron_vector = ", SUM_DE_Output_neuron_vector)
    
    temp_neg_out = [0 for i in range(len(neg_DE_Output_neuron_vector[0]))]
    for sample in neg_DE_Output_neuron_vector:
        for i in range(len(sample)):
            temp_neg_out[i] = temp_neg_out[i] + sample[i]
    SUM_neg_DE_Output_neuron_vector = deepcopy(temp_neg_out)
    #print("delta_E_function : SUM_neg_DE_Output_neuron_vector = ", SUM_neg_DE_Output_neuron_vector)
    
    
    SUM_DE_hidden_neuron_vector = SUM_DE_hidden_neuron_vector_function(DE_hidden_neuron_vector)
    SUM_neg_DE_hidden_neuron_vector = SUM_DE_hidden_neuron_vector_function(neg_DE_hidden_neuron_vector)
    
    #print("delta_E_function : SUM_DE_Output_neuron_vector DEBUG = ", SUM_DE_Output_neuron_vector)
    #print("delta_E_function : SUM_DE_hidden_neuron_vector DEBUG = ", SUM_DE_hidden_neuron_vector)
    #print("delta_E_function : SUM_neg_DE_Output_neuron_vector DEBUG = ", SUM_neg_DE_Output_neuron_vector)
    #print("delta_E_function : SUM_neg_DE_hidden_neuron_vector DEBUG = ", SUM_neg_DE_hidden_neuron_vector)            
                
    return SUM_DE_Output_neuron_vector, SUM_DE_hidden_neuron_vector, SUM_neg_DE_Output_neuron_vector, SUM_neg_DE_hidden_neuron_vector  

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
                    denom = denom + pow(neuron[j],2)
                    
        else:
            for neuron in DE_Output_neuron_vector[i]:
                denom = denom + pow(neuron,2)
                
            for neuron in DE_hidden_neuron_vector[i]:
                for j in range(len(neuron)):
                    denom = denom + pow(neuron[j],2)
    
    num = sqrt(num)
    denom = sqrt(denom)
    gamma = num/denom
    
    return gamma

def line_search_function_h_hidden(W_new_HLN1, W_new_out, h_hidden, h_out, data_input_Vgs, data_input_Vds, data_outputs):
    # check the line search method from prof's textbook (pg.60~)
    # etah_1 = min(=0) , etah_2 = max(= fixed number, user given)4

    count = 0
    # Prev_W_for_phi = []
    # W_hidden_for_phi = []
    # W_out_for_phi = []
    # W_hidden_delta = []
    # W_out_delta = []
    list_extender = [None,None,None,None,None,None,None]
    

    #my decision for abs(etah2 - etah1)
    zero_threshold = 0.00000001
    #print("line_search_function_h_hidden: W_new_HLN1 = ", W_new_HLN1)
    #print("line_search_function_h_hidden: h_hidden = ", h_hidden)
    #print("\n")
    
    #step 1
    etah1 = 0
    etah2 = 0.5
    etah3 = etah2 - 0.618*(etah2-etah1)
    etah4 = etah1 + 0.618*(etah2-etah1)
    #print("line_search_function_h_hidden: etah3 : ", etah3)
    #print("line_search_function_h_hidden: etah4 : ", etah4)
    #print("\n")
    
    W_hidden_original = W_new_HLN1
    #print("line_search_function_h_hidden: W_hidden_original : ", W_hidden_original)
    W_out_original = W_new_out
    #print("line_search_function_h_hidden: W_out_original : ", W_out_original)
    h_hidden_original = h_hidden
    #print("line_search_function_h_hidden: h_hidden_original : ", h_hidden_original)
    h_out_original = h_out
    #print("line_search_function_h_hidden: h_out_original : ", h_out_original)
    #print("\n")
    
    #step2
    for i in range(1000):
    
        phi3 = function_for_phi_calculation(W_hidden_original, W_out_original, h_hidden_original, h_out_original,  data_input_Vgs, data_input_Vds, data_outputs, etah3)
        #print("line_search_function_h_hidden: phi3 : ", phi3)
        
        phi4 = function_for_phi_calculation(W_hidden_original, W_out_original, h_hidden_original, h_out_original,  data_input_Vgs, data_input_Vds, data_outputs, etah4)
        #print("line_search_function_h_hidden: phi4 : ", phi4)
        #print("\n")
        
        #------------------------------------------------------------------#
        if (phi3 > phi4):
            etah1 = etah3
            etah3 = etah4
            etah4 = etah1 + 0.618*(etah2-etah1)

            phi4 = function_for_phi_calculation(W_hidden_original, W_out_original, h_hidden_original, h_out_original,  data_input_Vgs, data_input_Vds, data_outputs, etah4)
            
        else :
            etah2 = etah4
            etah4 = etah3
            etah3 = etah2 - 0.618*(etah2-etah1)
        
            phi3 = function_for_phi_calculation(W_hidden_original, W_out_original, h_hidden_original, h_out_original,  data_input_Vgs, data_input_Vds, data_outputs, etah3)
            
        if (abs(etah2 - etah1) < zero_threshold):
            etah = etah3
            break
        count = count + 1
              
    #print("count = ", count)
    return etah

def conjugate_gradient(user_Validation_error, W_HLN1, W_out, data_Tr_input_Vgs, data_Tr_input_Vds, data_Tr_output_Id, data_V_input_Vgs, data_V_input_Vds, data_V_output_Id, max_epoch):
    
    Training_error_collection = []
    num_epoch_iterated = 0
    W_new_HLN1 = [W_HLN1, None]
    W_new_out = [W_out, None]
    h_hidden = [None, None]
    h_out = [None, None]
    Delta_E_W_out  = [None, None]
    Delta_E_W_hidden  = [None, None]
    
    
    for epoch in range(max_epoch):
        #print("----------------------------------------------")
        print("conjugate_gradient : epoch = ", epoch)
        FF_error_Tr = error_checking(W_new_HLN1[0], W_new_out[0], data_Tr_input_Vgs, data_Tr_input_Vds, data_Tr_output_Id)
        FF_error_V = error_checking(W_new_HLN1[0], W_new_out[0], data_V_input_Vgs, data_V_input_Vds, data_V_output_Id)
        Training_error_collection.extend([FF_error_Tr])
        #print("conjugate_gradient: FF_error_Tr = ", FF_error_Tr)
        if ((FF_error_Tr < user_Validation_error) or (FF_error_V == FF_error_Tr)):
            break
        #print("conjugate_gradient: W_new_HLN1[0] initial = ", W_new_HLN1[0])
        #print("conjugate_gradient: W_new_out[0] initial = ", W_new_out[0])
        #print("\n")
        W_new_HLN1_epoch0 = deepcopy(W_new_HLN1[0])
        W_new_out_epoch0 = deepcopy(W_new_out[0])
        
        #
        DE_Output_neuron_vector, DE_hidden_neuron_vector, neg_DE_Output_neuron_vector, neg_DE_hidden_neuron_vector = delta_E_function(W_new_HLN1[0], W_new_out[0], data_Tr_input_Vgs, data_Tr_input_Vds, data_Tr_output_Id)
        Delta_E_W_out.insert(0, deepcopy(DE_Output_neuron_vector))
        Delta_E_W_hidden.insert(0, deepcopy(DE_hidden_neuron_vector))
        Delta_E_W_out.pop(-1)
        Delta_E_W_hidden.pop(-1)
        
        #print("conjugate_gradient: DE_Output_neuron_vector DEBUG = ", DE_Output_neuron_vector)
        #print("conjugate_gradient: DE_hidden_neuron_vector DEBUG = ", DE_hidden_neuron_vector)
        #print("\n")
    
        # for i in range(len(DE_Output_neuron_vector)):
        #     print("conjugate_gradient : DE_Output_neuron_vector [",i,"] = ", DE_Output_neuron_vector[i])
        # print("\n")
        # for i in range(len(DE_hidden_neuron_vector)):
        #     print("conjugate_gradient : DE_hidden_neuron_vector [",i,"] = ", DE_hidden_neuron_vector[i])
        # print("\n")
        # for i in range(len(neg_DE_Output_neuron_vector)):
        #         print("conjugate_gradient : neg_DE_Output_neuron_vector [",i,"] = ", neg_DE_Output_neuron_vector[i])
        # print("\n")
        # for i in range(len(neg_DE_hidden_neuron_vector)):
        #     print("conjugate_gradient : neg_DE_hidden_neuron_vector [",i,"] = ", neg_DE_hidden_neuron_vector[i])
        # print("\n")

        if (epoch == 0):
            
            h_hidden.insert(0, deepcopy(neg_DE_hidden_neuron_vector))
            h_out.insert(0, deepcopy(neg_DE_Output_neuron_vector))
            h_hidden.pop(-1)
            h_out.pop(-1)
            
            h_hidden_epoch0 = deepcopy(h_hidden[0])
            h_out_epoch0 = deepcopy(h_out[0])
            
            etah = line_search_function_h_hidden(W_new_HLN1_epoch0, W_new_out_epoch0, h_hidden_epoch0, h_out_epoch0, data_Tr_input_Vgs, data_Tr_input_Vds, data_Tr_output_Id)
            
            W_out_final, W_hidden_final = function_for_final_weight(W_new_HLN1_epoch0, W_new_out_epoch0, h_hidden_epoch0, h_out_epoch0,  data_Tr_input_Vgs, data_Tr_input_Vds, data_Tr_output_Id, etah)
            #print("conjugate_gradient: W_new_out DEBUG1 = ", W_new_out)
            #print("conjugate_gradient: W_new_HLN1 DEBUG1 = ", W_new_HLN1)
            #print("\n")
            
            W_new_HLN1.insert(0, deepcopy(W_hidden_final))
            W_new_out.insert(0, deepcopy(W_out_final))
            #print("conjugate_gradient: W_new_out DEBUG2 = ", W_new_out)
            #print("conjugate_gradient: W_new_HLN1 DEBUG2 = ", W_new_HLN1)
            #print("\n")
            
            W_new_HLN1.pop(-1)
            W_new_out.pop(-1)
            # print("conjugate_gradient: W_new_out = ", W_new_out)
            # print("conjugate_gradient: W_new_HLN1 = ", W_new_HLN1)
            # print("\n")
            
        else:
            # print("conjugate_gradient: DE_Output_neuron_vector (epoch >= ",epoch,") = ", DE_Output_neuron_vector)
            # print("conjugate_gradient: DE_hidden_neuron_vector (epoch >= ",epoch,") = ", DE_hidden_neuron_vector)
            # print("\n")
            
            # print("conjugate_gradient: Delta_E_W_out (epoch >= ",epoch,") = ", Delta_E_W_out)
            # print("conjugate_gradient: Delta_E_W_hidden (epoch >= ",epoch,") = ", Delta_E_W_hidden)
            # print("\n")
            
            gamma = gamma_function(Delta_E_W_out, Delta_E_W_hidden)
            # print("conjugate_gradient: gamma (epoch >= ",epoch,") = ", gamma)
            # print("\n")
            # print("conjugate_gradient: h_out (epoch >= ",epoch,") = ", h_out)
            # print("conjugate_gradient: h_hidden (epoch >= ",epoch,") = ", h_hidden)
            # print("conjugate_gradient: neg_DE_Output_neuron_vector (epoch >= ",epoch,") = ", neg_DE_Output_neuron_vector)
            # print("conjugate_gradient: neg_DE_hidden_neuron_vector (epoch >= ",epoch,") = ", neg_DE_hidden_neuron_vector)
            # print("\n")
            
            h_out.pop(-1)
            h_hidden.pop(-1)
            # print("conjugate_gradient: h_out (epoch >= ",epoch,") DEBUG1 = ", h_out)
            # print("conjugate_gradient: h_hidden (epoch >= ",epoch,") DEBUG1 = ", h_hidden)
            # print("\n")
            
            delta_h_out, delta_h_hidden = function_gamma_x_prev_h_epoch(h_out[0], h_hidden[0], gamma)
            # print("conjugate_gradient: delta_h_out (epoch >= ",epoch,") = ", delta_h_out)
            # print("conjugate_gradient: delta_h_hidden (epoch >= ",epoch,") = ", delta_h_hidden)
            # print("\n")
            
            final_h_out, final_h_hidden = function_sum_neg_deltaE_delta_h(delta_h_out, delta_h_hidden, neg_DE_Output_neuron_vector, neg_DE_hidden_neuron_vector)
            
            # print("conjugate_gradient: final_h_out (epoch >= ",epoch,") = ", final_h_out)
            # print("conjugate_gradient: final_h_hidden (epoch >= ",epoch,") = ", final_h_hidden)
            # print("\n")
            
            h_out.insert(0,deepcopy(final_h_out))
            h_hidden.insert(0,deepcopy(final_h_hidden))
            # print("conjugate_gradient: h_out DEBUG2 (epoch >= ",epoch,") = ", h_out)
            # print("conjugate_gradient: h_hidden DEBUG2 (epoch >= ",epoch,") = ", h_hidden)
            # print("\n")
            
            etah = line_search_function_h_hidden(W_new_HLN1[0], W_new_out[0], h_hidden[0], h_out[0], data_Tr_input_Vgs, data_Tr_input_Vds, data_Tr_output_Id)
            W_out_final, W_hidden_final = function_for_final_weight(W_new_HLN1[0],  W_new_out[0], h_hidden[0], h_out[0],  data_Tr_input_Vgs, data_Tr_input_Vds, data_Tr_output_Id, etah)
            # print("conjugate_gradient: W_out_final (epoch >= ",epoch,") = ", W_out_final)
            # print("conjugate_gradient: W_hidden_final (epoch >= ",epoch,") = ", W_hidden_final)
            # print("\n")
            
            W_new_out.insert(0, W_out_final)
            W_new_HLN1.insert(0, W_hidden_final)
            # print("conjugate_gradient: W_new_out (epoch >= ",epoch,") = ", W_new_out)
            # print("conjugate_gradient: W_new_HLN1 (epoch >= ",epoch,") = ", W_new_HLN1)
            # print("\n")
            
            W_new_out.pop(-1)
            W_new_HLN1.pop(-1)
            # print("conjugate_gradient: W_new_out (epoch >= ",epoch,") = ", W_new_out)
            # print("conjugate_gradient: W_new_HLN1 (epoch >= ",epoch,") = ", W_new_HLN1)
            # print("\n")
            
            
        num_epoch_iterated = num_epoch_iterated + 1
        #print("conjugate_gradient: num_epoch_iterated (epoch >= ",epoch,") = ", num_epoch_iterated)    
            
    return W_new_out[0], W_new_HLN1[0], num_epoch_iterated, Training_error_collection

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

# Converting the strings of string elememts of each sample into float, what float? because there are negative values
for sample in data_Tr:
    for i in range(len(sample)):
        sample[i] = float(sample[i])
        
for sample2 in data_V:
    for i in range(len(sample)):
        sample2[i] = float(sample2[i])

# ---------------------------------------------------
# Data Generation and scaling of Training set
data_Tr_input_Vgs = []
data_Tr_input_Vds = []
data_Tr_output_Id = []
for sample in data_Tr:
    for j in range(len(sample)):
        if (j == (len(sample)-3)):
            data_Tr_input_Vgs.append(sample[j])
        if (j == (len(sample)-2)):
            data_Tr_input_Vds.append(sample[j])
        if (j == (len(sample)-1)):
            data_Tr_output_Id.append(sample[j])

# ---------------------------------------------------
# Data Generation and scaling of Validation set
data_V_input_Vgs = []
data_V_input_Vds = []
data_V_output_Id = []
for sample2 in data_V:
    for j in range(len(sample2)):
        if (j == (len(sample2)-3)):
            data_V_input_Vgs.append(sample2[j])
        if (j == (len(sample2)-2)):
            data_V_input_Vds.append(sample2[j])
        if (j == (len(sample2)-1)):
            data_V_output_Id.append(sample2[j])

print("data_Tr_input_Vgs list is : ", data_Tr_input_Vgs)
print("data_Tr_input_Vds list is : ", data_Tr_input_Vds)
print("data_Tr_output_Id list is : ", data_Tr_output_Id)
print("\n")
print("data_V_input_Vgs list is : ", data_V_input_Vgs)
print("data_V_input_Vds list is : ", data_V_input_Vds)
print("data_V_output_Id list is : ", data_V_output_Id)
print("\n")

Ali_input_num = 2

temp_input = Ali_input_num

# Initialize the weight values when a number was typed by an user in GUI
hidden1 = 3
if (hidden1 > 0):              
    temp_W_HLN1 = [[2*random()-1 for i in range(temp_input + 1)] for i in range(hidden1)] # Initializing random weights and biases in a range of -1 to 1
print("temp_W_HLN1 : " , temp_W_HLN1)

W_HLN1 = deepcopy(temp_W_HLN1)
W_out = []
#for neuron1 in temp_W_HLN1:
#    for i in range(len(neuron1)):
#        neuron1[i] = round(neuron1[i], 4)
#    W_HLN1.append(neuron1)

W_out = [2*random()-1 for i in range(len(W_HLN1) + 1) ]
#for h in range(len(W_out)):
#    W_out[h] = round(W_out[h],4)

# Hidden neurons will be activated with the sigmoid function
# Output neuron will be activated with the linear function
print("W_HLN1 : " , W_HLN1)
print("W_out : " , W_out)
print("\n")

max_epoch = 500;
user_Validation_error = 0.1;
[W_new_out, W_new_HLN1, num_epoch_iter, Training_error_collection] = conjugate_gradient(user_Validation_error, W_HLN1, W_out, data_Tr_input_Vgs, data_Tr_input_Vds, data_Tr_output_Id, data_V_input_Vgs, data_V_input_Vds, data_V_output_Id, max_epoch)

print("W_new_out : " , W_new_out)
print("W_new_HLN1 : " , W_new_HLN1)
print("num_epoch_iter : " , num_epoch_iter)
print("Training_error_collection : " , Training_error_collection)

#-----------------------NOTE----------------------------
#The structure of the ANN with one hidden layer
#                    Id
#        # of Hidden Neurons here
#                Vgs     Vds






