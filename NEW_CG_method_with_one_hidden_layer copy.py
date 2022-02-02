from copy import deepcopy
from operator import index
from re import I
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
    SUM_HL1_comp = np.add(HL1_comp, B_HL1_new)
    #print("feedforward_comp: SUM_HL1_comp = ", np.array(SUM_HL1_comp))
    #print("\n")
    
    # Applying the sigmoid activation function to activate all the elements (neuron outputs) for the data set
    Activated_SUM_HL1_comp = np.array([[transfer_sigmoid(x) for x in sample] for sample in deepcopy(SUM_HL1_comp)])
    #print("feedforward_comp: Activated_SUM_HL1_comp = ", Activated_SUM_HL1_comp)
    #print("\n")
    
    final_calculated_outputs = np.transpose(np.add(np.dot(W_out_new, Activated_SUM_HL1_comp.T), B_out_new))
    #print("feedforward_comp: final_calculated_outputs = ", final_calculated_outputs)
    #print("\n")
    
    return final_calculated_outputs, Activated_SUM_HL1_comp


def Phi_function_HL1(W_out_new, W_HL1_new, B_out_new, B_HL1_new,
                            h_W_out, h_W_hidden, h_B_HL1_out, h_B_HL1_hidden,
                            inputs_Tr, data_outputs_Tr, etah):
    
    # About: This function is for calculating an error with the etah value from line minimization
    # print("---------------------------------------------------------------------")
    # print("Phi_function_HL1: W_out_new = ", W_out_new)
    # print("Phi_function_HL1: B_out_new = ", B_out_new)
    # print("\n")
    # print("Phi_function_HL1: W_HL1_new = ", W_HL1_new)
    # print("Phi_function_HL1: B_HL1_new = ", B_HL1_new)
    # print("\n")
    # print("Phi_function_HL1: h_W_out = ", h_W_out)
    # print("Phi_function_HL1: h_W_hidden = ", h_W_hidden)
    # print("\n")
    # print("Phi_function_HL1: h_B_HL1_out = ", h_B_HL1_out)
    # print("Phi_function_HL1: h_B_HL1_hidden = ", h_B_HL1_hidden)
    # print("\n")
    # print("Phi_function_HL1: etah = ", etah)
    # print("\n")
    
    delta_h_W_out = np.multiply(h_W_out, etah)
    delta_h_B_out = np.multiply(h_B_HL1_out, etah)
    delta_h_W_HL1 = np.multiply(h_W_hidden, etah)
    delta_h_B_HL1 = np.multiply(h_B_HL1_hidden, etah)
    
    # print("Phi_function_HL1: delta_h_W_out = ", delta_h_W_out)
    # print("Phi_function_HL1: delta_h_W_HL1 = ", delta_h_W_HL1)
    # print("\n")
    # print("Phi_function_HL1: delta_h_B_out = ", delta_h_B_out)
    # print("Phi_function_HL1: delta_h_B_HL1 = ", delta_h_B_HL1)
    # print("\n")
    
    W_out_phi = np.add(W_out_new, delta_h_W_out)
    B_out_phi = np.add(B_out_new, delta_h_B_out)
    W_HL1_phi = np.add(W_HL1_new, delta_h_W_HL1)
    B_HL1_phi = np.add(B_HL1_new, delta_h_B_HL1)
    
    # print("Phi_function_HL1: W_out_phi = ", W_out_phi)
    # print("Phi_function_HL1: B_out_phi = ", B_out_phi)
    # print("\n")
    # print("Phi_function_HL1: W_HL1_phi = ", W_HL1_phi)
    # print("Phi_function_HL1: B_HL1_phi = ", B_HL1_phi)
    # print("\n")

    # print("Phi_function_HL1: inputs_Tr = ", inputs_Tr)
    # print("Phi_function_HL1: data_outputs_Tr = ", data_outputs_Tr)
    # print("\n")
    
    calculated_outputs, dummy = feedforward_comp(W_HL1_phi, B_HL1_phi, W_out_phi, B_out_phi, inputs_Tr)
    phi = error_checking(calculated_outputs, data_outputs_Tr)
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
    
    # print("-----------------------------------------------------------------")
    # print("error_checking: calculated_outputs = ", calculated_outputs)
    # print("error_checking: data_outputs = ", data_outputs)
    # print("\n")
    
    # The error equation (Loss function) => 0.5*(calculated - data)^2
    error = np.multiply(np.square(np.subtract(calculated_outputs, data_outputs)), 0.5)
    #print("error_checking: error = ", error)
    #print("\n")
    Avg_error = np.sum(error)/len(data_outputs)
    #print("error_checking: Avg_error = ", Avg_error)
    #print("\n")
    
    return Avg_error

def delta_W_HL1_computation(activated_hidden1_outputs, data_inputs, W_HL1_size):
    
    # About: To find the delta E of Weights in the hidden 1 layer. Since there are 12 samples to hanle
    #        array/matrix will be computed for one sample each for simplicity
    
    # print("-----------------------------------------------------------------")
    # print("delta_W_HL1_computation: activated_hidden1_outputs = ", activated_hidden1_outputs)
    # print("delta_W_HL1_computation: data_inputs = ", data_inputs)
    # print("delta_W_HL1_computation: W_HL1_size = ", W_HL1_size)
    # print("\n")
    
    # need to make a matrix that is the same size matrix as W_HL1 and the matrix is the delta E function for each sample
    delta_w_HL1 = np.array([[pow(activated_hidden1_outputs[row], 2)*(1 - activated_hidden1_outputs[row])*data_inputs[col] for col in range(W_HL1_size[1])] for row in range(W_HL1_size[0])])
    # print("delta_W_HL1_computation: delta_w_HL1 = ", delta_w_HL1)
    # print("\n")
    
    return delta_w_HL1

def de_E_function_HL1(W_HL1_new, B_HL1_new, W_out_new, B_out_new, calculated_outputs_Tr, activated_hidden1_outputs, data_inputs, data_outputs):

    # About: With the present epoch weights, derivative of output weight and hidden weight will be determined
    # State the derivate of a function for each neuron weight/bias
    
    # print("-----------------------------------------------------------------")
    # print("de_E_function_HL1: W_HL1_new = ", W_HL1_new)
    # print("de_E_function_HL1: B_HL1_new = ", B_HL1_new)
    # print("\n")
    # print("de_E_function_HL1: W_out_new = ", W_out_new)
    # print("de_E_function_HL1: B_out_new = ", B_out_new)
    # print("\n")
    # print("de_E_function_HL1: calculated_outputs_Tr = ", calculated_outputs_Tr)
    # print("de_E_function_HL1: activated_hidden1_outputs = ", activated_hidden1_outputs)
    # print("\n")
    # print("de_E_function_HL1: data_inputs = ", data_inputs)
    # print("de_E_function_HL1: data_outputs = ", data_outputs)
    # print("\n")
    
    Diff_Calc_Measure = np.subtract(calculated_outputs_Tr, data_outputs)
    #print("de_E_function_HL1: Diff_Calc_Measure = ", Diff_Calc_Measure)
    #print("\n")
    
    # -------------------------------- Output neuron below ----------------------------------------
    # -------------------------------- Weight and Bias --------------------------------------------    
    
    SUM_de_B_out = np.sum(Diff_Calc_Measure)
    #print("de_E_function_HL1: SUM_de_B_out = ", SUM_de_B_out)
    #print("\n")
    
    de_w_out = np.multiply(activated_hidden1_outputs, Diff_Calc_Measure)
    SUM_de_w_out = de_w_out.sum(axis=0)
    #print("de_E_function_HL1: de_w_out = ", de_w_out)
    #print("de_E_function_HL1: SUM_de_w_out = ", SUM_de_w_out)
    #print("\n")
    
    

    # -------------------------------- Hidden neurons below ----------------------------------------
    # Delta E vector for the weights and the bias of the output neuron with entire data set(12 in total)
    # Prev_#### means only the vector portion, not with the multiplication of (calc-measure)
    
    de_B_HL1 = np.array([[pow(x,2)*(1 - x) for x in sample] for sample in activated_hidden1_outputs])
    SUM_de_B_HL1 = de_B_HL1.sum(axis=0)
    #print("delta_E_function_HL1: de_B_HL1 = ", de_B_HL1)
    #print("delta_E_function_HL1: SUM_de_B_HL1 = ", SUM_de_B_HL1)
    #print("\n")

    # Find the matrix size of the hidden layer 1 weights
    W_HL1_size = np.shape(W_HL1_new)
    #print("delta_E_function_HL1: W_HL1_size = ", W_HL1_size)
    #print("\n")
    
    # Since a huge collection of arrays will be generated, for simplicity, created a function to allocate each data sample for finding delta dE/dw
    de_w_HL1 = np.array([None for i in range(len(activated_hidden1_outputs))])
    for i in range(len(activated_hidden1_outputs)):
        de_w_HL1[i] = delta_W_HL1_computation(activated_hidden1_outputs[i], data_inputs[i], W_HL1_size)
    #print("delta_E_function_HL1: de_w_HL1 = ", de_w_HL1)
    #print("\n")
    
    SUM_de_w_HL1 = de_w_HL1.sum(axis=0)
    #print("delta_E_function_HL1: SUM_de_w_HL1 = ", SUM_de_w_HL1)
    #print("\n")
                   
    return SUM_de_w_out, SUM_de_B_out, SUM_de_w_HL1, SUM_de_B_HL1  

def gamma_function(Delta_E_W_out, Delta_E_W_hidden, Delta_B_out, Delta_B_hidden):

    print("--------------------------------------------------------------------")
    # The length of DE_out and DE_hidden is 2
    # The first element of both is the present epoch and second is the previous epoch
    
    # print("gamma_function : Delta_E_W_out = ", Delta_E_W_out)
    # print("gamma_function : Delta_E_W_hidden = ", Delta_E_W_hidden)
    # print("gamma_function : Delta_B_out = ", Delta_B_out)
    # print("gamma_function : Delta_B_hidden = ", Delta_B_hidden)
    # print("\n")
    
    num = 0         # initialize the numerator and the denominator to zero
    denom = 0

    for i in range(len(Delta_E_W_out)):
        if (i == 0):
            num = np.sum(np.square(Delta_E_W_out[i])) + np.sum(np.square(Delta_E_W_hidden[i])) + np.sum(np.square(Delta_B_out[i])) + np.sum(np.square(Delta_B_hidden[i]))   
        else:
            denom = np.sum(np.square(Delta_E_W_out[i])) + np.sum(np.square(Delta_E_W_hidden[i])) + np.sum(np.square(Delta_B_out[i])) + np.sum(np.square(Delta_B_hidden[i]))
    
    # print("gamma_function : num = ", num)
    # print("gamma_function : denom = ", denom)
    # print("\n")

    gamma = num/denom
    

    #print("gamma_function : gamma = ", gamma)
    #print("\n")
    
    return gamma


def line_search_function_h_hidden(W_out_new, W_HL1_new, B_out_new, B_HL1_new,
                                    h_W_out, h_W_hidden, h_B_out, h_B_hidden,
                                    inputs_Tr, data_outputs_Tr):
    
    #print("--------------------------------------------------------------------")
    # The length of DE_out and DE_hidden is 2
    # The first element of both is the present epoch and second is the previous epoch
    
    # print("line_search_function_h_hidden : W_out_new = ", W_out_new)
    # print("line_search_function_h_hidden : B_out_new = ", B_out_new)
    # print("\n")
    # print("line_search_function_h_hidden : W_HL1_new = ", W_HL1_new)
    # print("line_search_function_h_hidden : B_HL1_new = ", B_HL1_new)
    # print("\n")
    # print("line_search_function_h_hidden : h_W_out = ", h_W_out)
    # print("line_search_function_h_hidden : h_W_hidden = ", h_W_hidden)
    # print("\n")
    # print("line_search_function_h_hidden : h_B_out = ", h_B_out)
    # print("line_search_function_h_hidden : h_B_hidden = ", h_B_hidden)
    # print("\n")
    
    # check the line search method from prof's textbook (pg.60~)
    # etah_1 = min(=0) , etah_2 = max(= fixed number, user given)4

    count = 0
    zero_threshold = 0.00000001 # seven zeros originally
    
    # step 1
    etah1 = 0
    etah2 = 0.5
    etah3 = etah2 - 0.618*(etah2-etah1)
    etah4 = etah1 + 0.618*(etah2-etah1)

    #step2
    for i in range(1000):
        
        # Phi is the calculated error with W + etah*h
        phi3 = Phi_function_HL1(W_out_new, W_HL1_new, B_out_new, B_HL1_new,
                            h_W_out, h_W_hidden, h_B_out, h_B_hidden,
                            inputs_Tr, data_outputs_Tr, etah3)
        phi4 = Phi_function_HL1(W_out_new, W_HL1_new, B_out_new, B_HL1_new,
                            h_W_out, h_W_hidden, h_B_out, h_B_hidden,
                            inputs_Tr, data_outputs_Tr, etah4)
        #print("line_search_function_h_hidden : phi3 = ", phi3)
        #print("line_search_function_h_hidden : phi4 = ", phi4)
        #print("\n")
        
        #------------------------------------------------------------------#
        if (phi3 > phi4):
            etah1 = etah3
            etah3 = etah4
            etah4 = etah1 + 0.618*(etah2-etah1)

            phi4 = Phi_function_HL1(W_out_new, W_HL1_new, B_out_new, B_HL1_new,
                            h_W_out, h_W_hidden, h_B_out, h_B_hidden,
                            inputs_Tr, data_outputs_Tr, etah4)
            
        else :
            etah2 = etah4
            etah4 = etah3
            etah3 = etah2 - 0.618*(etah2-etah1)
        
            phi3 = Phi_function_HL1(W_out_new, W_HL1_new, B_out_new, B_HL1_new,
                            h_W_out, h_W_hidden, h_B_out, h_B_hidden,
                            inputs_Tr, data_outputs_Tr, etah3)
            
        if (abs(etah2 - etah1) < zero_threshold):
            etah = etah3
            break
        count = count + 1
              
    #print("line_search_function_h_hidden : count = ", count)
    #print("line_search_function_h_hidden : etah = ", etah)
    #print("line_search_function_h_hidden : count = ", count)
    #print("\n")
    return etah

def Conjugate_Gradient_HL1(user_Validation_error, W_HL1, B_HL1, W_out, B_out, inputs_Tr, data_outputs_Tr, inputs_V, data_outputs_V, inputs_Te, data_outputs_Te, max_epoch):

    Training_error_collection = []
    num_epoch_iterated = 0
    
    W_out_new = [W_out, None]
    W_HL1_new = [W_HL1, None]
    
    B_out_new = [B_out, None]
    B_HL1_new = [B_HL1, None]
    
    h_W_out = [None, None]
    h_W_hidden = [None, None]
    
    h_B_out = [None, None]
    h_B_hidden = [None, None]
    
    Delta_E_W_out  = [None, None]
    Delta_E_W_hidden  = [None, None]
    
    Delta_B_out  = [None, None]
    Delta_B_hidden  = [None, None]
    
    
    for epoch in range(max_epoch):
        #print("------------------------------------------------------------------- ")
        #print("Conjugate_Gradient_HL1 : epoch = ", epoch)
        calculated_outputs_Tr, activated_hidden1_outputs_Tr = feedforward_comp(W_HL1_new[0], B_HL1_new[0], W_out_new[0], B_out_new[0], inputs_Tr)
        FF_error_Tr = error_checking(calculated_outputs_Tr, data_outputs_Tr)
        
        calculated_outputs_V, dummy = feedforward_comp(W_HL1_new[0], B_HL1_new[0], W_out_new[0], B_out_new[0], inputs_V)
        FF_error_V = error_checking(calculated_outputs_V, data_outputs_V)
        #print("Conjugate_Gradient_HL1 : FF_error_Tr = ", FF_error_Tr)
        #print("Conjugate_Gradient_HL1 : FF_error_V = ", FF_error_V)
        #print("\n")
        
        Training_error_collection.extend([FF_error_Tr])
        if (FF_error_V < user_Validation_error):
            break

        SUM_de_w_out, SUM_de_B_out, SUM_de_w_HL1, SUM_de_B_HL1 = de_E_function_HL1(W_HL1_new[0], B_HL1_new[0], 
                                                                                   W_out_new[0], B_out_new[0], 
                                                                                   calculated_outputs_Tr, activated_hidden1_outputs_Tr, 
                                                                                   inputs_Tr, data_outputs_Tr)
        
        NEG_SUM_de_w_out = np.multiply(SUM_de_w_out, -1)
        NEG_SUM_de_B_out = np.multiply(SUM_de_B_out, -1)
        NEG_SUM_de_w_HL1 = np.multiply(SUM_de_w_HL1, -1)
        NEG_SUM_de_B_HL1 = np.multiply(SUM_de_B_HL1, -1)
        # print("Conjugate_Gradient_HL1 : SUM_de_w_out = ", SUM_de_w_out)
        # print("Conjugate_Gradient_HL1 : SUM_de_B_out = ", SUM_de_B_out)
        # print("Conjugate_Gradient_HL1 : SUM_de_w_HL1 = ", SUM_de_w_HL1)
        # print("Conjugate_Gradient_HL1 : SUM_de_B_HL1 = ", SUM_de_B_HL1)
        # print("\n")
        
        Delta_E_W_out.insert(0, deepcopy(SUM_de_w_out))
        Delta_E_W_hidden.insert(0, deepcopy(SUM_de_w_HL1))
        Delta_B_out.insert(0, deepcopy(SUM_de_w_HL1))
        Delta_B_hidden.insert(0, deepcopy(SUM_de_B_HL1))
        
        # print("Conjugate_Gradient_HL1 : Delta_E_W_out = ", Delta_E_W_out)
        # print("Conjugate_Gradient_HL1 : Delta_E_W_hidden = ", Delta_E_W_hidden)
        # print("Conjugate_Gradient_HL1 : Delta_B_out = ", Delta_B_out)
        # print("Conjugate_Gradient_HL1 : Delta_B_hidden = ", Delta_B_hidden)
        # print("\n")
        
        Delta_E_W_out.pop(-1)
        Delta_E_W_hidden.pop(-1)
        Delta_B_out.pop(-1)
        Delta_B_hidden.pop(-1)
        
        # print("Conjugate_Gradient_HL1 : Delta_E_W_out = ", Delta_E_W_out)
        # print("Conjugate_Gradient_HL1 : Delta_E_W_hidden = ", Delta_E_W_hidden)
        # print("Conjugate_Gradient_HL1 : Delta_B_out = ", Delta_B_out)
        # print("Conjugate_Gradient_HL1 : Delta_B_hidden = ", Delta_B_hidden)
        # print("\n")
    
        if (epoch == 0):
            
            W_out_new_epoch0 = deepcopy(W_out_new[0])
            W_HL1_new_epoch0 = deepcopy(W_HL1_new[0])
            B_out_new_epoch0 = deepcopy(B_out_new[0])
            B_HL1_new_epoch0 = deepcopy(B_HL1_new[0])
            
            h_W_out.insert(0, deepcopy(NEG_SUM_de_w_out))
            h_W_hidden.insert(0, deepcopy(NEG_SUM_de_w_HL1))
            h_B_out.insert(0, deepcopy(NEG_SUM_de_B_out))
            h_B_hidden.insert(0, deepcopy(NEG_SUM_de_B_HL1))
            
            h_W_out.pop(-1)
            h_W_hidden.pop(-1)
            h_B_out.pop(-1)
            h_B_hidden.pop(-1)
            
            h_W_out_epoch0 = deepcopy(h_W_out[0])
            h_W_hidden_epoch0 = deepcopy(h_W_hidden[0])
            h_B_out_epoch0 = deepcopy(h_B_out[0])
            h_B_hidden_epoch0 = deepcopy(h_B_hidden[0])
            
            etah = line_search_function_h_hidden(W_out_new_epoch0, W_HL1_new_epoch0, B_out_new_epoch0, B_HL1_new_epoch0,
                                                 h_W_out_epoch0, h_W_hidden_epoch0, h_B_out_epoch0, h_B_hidden_epoch0,
                                                 inputs_Tr, data_outputs_Tr)

            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): etah = ", etah)
            
            W_out_final_epoch0 = np.add(W_out_new_epoch0, np.multiply(h_W_out_epoch0,etah))
            W_HL1_final_epoch0 = np.add(W_HL1_new_epoch0, np.multiply(h_W_hidden_epoch0,etah))
            B_out_final_epoch0 = np.add(B_out_new_epoch0, np.multiply(h_B_out_epoch0,etah))
            B_HL1_final_epoch0 = np.add(B_HL1_new_epoch0, np.multiply(h_B_hidden_epoch0,etah))
            
            W_out_new.insert(0, deepcopy(W_out_final_epoch0))
            W_HL1_new.insert(0, deepcopy(W_HL1_final_epoch0))
            B_out_new.insert(0, deepcopy(B_out_final_epoch0))
            B_HL1_new.insert(0, deepcopy(B_HL1_final_epoch0))
            
            W_out_new.pop(-1)
            W_HL1_new.pop(-1)
            B_out_new.pop(-1)
            B_HL1_new.pop(-1)
            
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): W_out_new = ", W_out_new)
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): W_HL1_new = ", W_HL1_new)
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): B_out_new = ", B_out_new)
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): B_HL1_new = ", B_HL1_new)
            print("\n")
            
        else: # epoch > 1
            
            gamma = gamma_function(Delta_E_W_out, Delta_E_W_hidden, Delta_B_out, Delta_B_hidden)
            
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): h_W_out DEBUG = ", h_W_out)
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): h_W_hidden DEBUG = ", h_W_hidden)
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): h_B_out DEBUG = ", h_B_out)
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): h_B_hidden DEBUG = ", h_B_hidden)
            print("\n")
            
            h_W_out.pop(-1)
            h_W_hidden.pop(-1)
            h_B_out.pop(-1)
            h_B_hidden.pop(-1)
            
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): h_W_out = ", h_W_out)
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): h_W_hidden = ", h_W_hidden)
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): h_B_out = ", h_B_out)
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): h_B_hidden = ", h_B_hidden)
            print("\n")
            
            etah = line_search_function_h_hidden(W_out_new[0], W_HL1_new[0], B_out_new[0], B_HL1_new[0],
                                                 h_W_out[0], h_W_hidden[0], h_B_out[0], h_B_hidden[0],
                                                 inputs_Tr, data_outputs_Tr)
            
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): etah = ", etah)
            #print(" Conjugate_Gradient_HL1: gamma (epoch = ",epoch,") = ", gamma)
            #print(" Conjugate_Gradient_HL1: etah (epoch = ",epoch,") = ", etah)
            #print("\n")
            
            # h(epoch) = -delta_E + " gamma * h(epoch -1) ", Computation for double quote
            delta_h_W_out = np.multiply(h_W_out[1], gamma)
            delta_h_W_hidden = np.multiply(h_W_hidden[1], gamma)
            delta_B_W_out = np.multiply(h_B_out[1], gamma)
            delta_B_W_hidden = np.multiply(h_B_hidden[1], gamma)
            
            # " h(epoch) = -delta_E + gamma * h(epoch -1) "
            final_delta_h_W_out = np.array(NEG_SUM_de_w_out) + np.array(delta_h_W_out)
            final_delta_h_W_hidden = np.array(NEG_SUM_de_w_HL1) + np.array(delta_h_W_hidden)
            final_delta_B_W_out = np.array(NEG_SUM_de_B_out) + np.array(delta_B_W_out)
            final_delta_B_W_hidden = np.array(NEG_SUM_de_B_HL1) + np.array(delta_B_W_hidden)

            # Inserting the updated, new h vectors to the first element
            h_W_out.insert(0,deepcopy(final_delta_h_W_out))
            h_W_hidden.insert(0,deepcopy(final_delta_h_W_hidden))
            h_B_out.insert(0,deepcopy(final_delta_B_W_out))
            h_B_hidden.insert(0,deepcopy(final_delta_B_W_hidden))

            # W = W + delta_W
            W_out_final = np.add(W_out_new[0], np.multiply(h_W_out[1], etah))
            W_hidden_final = np.add(W_HL1_new[0], np.multiply(h_W_hidden[1], etah))
            B_out_final = np.add(B_out_new[0], np.multiply(h_B_out[1], etah))
            B_hidden_final  = np.add(B_HL1_new[0], np.multiply(h_B_hidden[1], etah))
            
            
            W_out_new.insert(0, deepcopy(W_out_final))
            W_HL1_new.insert(0, deepcopy(W_hidden_final))
            B_out_new.insert(0, deepcopy(B_out_final))
            B_HL1_new.insert(0, deepcopy(B_hidden_final))
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): W_out_new = ", W_out_new)
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): W_HL1_new = ", W_HL1_new)
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): B_out_new = ", B_out_new)
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): B_HL1_new = ", B_HL1_new)
            print("\n")
            
            W_out_new.pop(-1)
            W_HL1_new.pop(-1)
            B_out_new.pop(-1)
            B_HL1_new.pop(-1)
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): W_out_new = ", W_out_new)
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): W_HL1_new = ", W_HL1_new)
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): B_out_new = ", B_out_new)
            print("Conjugate_Gradient_HL1 (epoch = ",epoch,"): B_HL1_new = ", B_HL1_new)
            print("\n")

        num_epoch_iterated = num_epoch_iterated + 1
        #print("conjugate_gradient: num_epoch_iterated (epoch >= ",epoch,") = ", num_epoch_iterated)    
    calculated_outputs_Te, dummy = feedforward_comp(W_HL1_new[0], B_HL1_new[0], W_out_new[0], B_out_new[0], inputs_Te)
    FF_error_Te = error_checking(calculated_outputs_Te, data_outputs_Te)

    return W_HL1_new[0], B_HL1_new[0], W_out_new[0], B_out_new[0], num_epoch_iterated, Training_error_collection, FF_error_Te

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
Ali_HL1_neuron = 1
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

# print("data_Tr = ", data_Tr)
# print("data_V = ", data_V)
# print("data_Te = ", data_Te)
# print("\n")

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

# print("data_Tr_inputs = ", data_Tr_inputs)
# print("data_Tr_outputs = ", data_Tr_outputs)
# print("\n")
# print("data_V_inputs = ", data_V_inputs)
# print("data_V_outputs = ", data_V_outputs)
# print("\n")
# print("data_Te_inputs = ", data_Te_inputs)
# print("data_Te_outputs = ", data_Te_outputs)
# print("\n")

# Initialize the weight values when a number was typed by an user in GUI

max_epoch = 3;
user_Validation_error = 0.001;



if (Ali_num_HL == 1):
    W_HL1 = np.random.rand(Ali_HL1_neuron, Ali_num_inputs)
    B_HL1 = np.random.rand(Ali_HL1_neuron)
    W_out = np.random.rand(Ali_num_outputs, len(W_HL1))
    B_out = np.random.rand(Ali_num_outputs)
    print("W_HL1 START = ", W_HL1)
    print("B_HL1 START = ", B_HL1)
    print("W_out START = ", W_out)
    print("B_out START = ", B_out)
    print("\n")
    [W_HL1_new, B_HL1_new, W_out_new, B_out_new, num_epoch_iter, Training_error_collection, FF_error_Te] = Conjugate_Gradient_HL1(user_Validation_error, W_HL1, B_HL1, W_out, B_out, data_Tr_inputs, data_Tr_outputs, data_V_inputs, data_V_outputs, data_Te_inputs, data_Te_outputs, max_epoch)
elif (Ali_num_HL == 2):
    W_HL1 = np.random.rand(len(data_Te), Ali_num_inputs)
    W_HL2 = np.random.rand(len(data_Te), Ali_num_inputs)
print("W_HL1_new END = ", W_HL1_new)
print("B_HL1_new END = ", B_HL1_new)
print("W_out_new END = ", W_out_new)
print("B_out_new END = ", B_out_new)
print("\n")

print("num_epoch_iter END = ", num_epoch_iter)
print("Training_error_collection END = ", Training_error_collection)
print("FF_error_Te END = ", FF_error_Te)
print("\n")