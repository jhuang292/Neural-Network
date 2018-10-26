import arff
import sys
import math
import random
import numpy as np


# n-fold stratified cross validation
def _n_fold_stratified_cross_validation(n, data):
 
    # Cross validation dictionary
    cvDic = {}
    cvDic['TRAIN'] = list()
    cvDic['TEST'] = list()
    
    # Get the size of each fold from the end of dataset
    if(n == 1):
      print("Wrong number for n-Folds")
    else:
       # size list to store the size of each cross validation fold
       sizeList = list()
       # Total data set size counter for splitting dataset
       totalSize = len(data['data'])
       while(n != 1):
            
            pSize = int(totalSize/n)
            sizeList.insert(0, pSize)
            totalSize -= pSize
            n -= 1
       sizeList.insert(0,totalSize)
   
    # Cross validation to get the index of each data set lines
    indexList = list()
    # Index counter for the data set
    index = 0
    for item in sizeList:
        partitionList = list()
        itemCounter = item
        while(index != len(data['data']) and itemCounter != 0):
             partitionList.append(index)
             index += 1
             itemCounter -= 1
        indexList.append(partitionList)
                                                                           
    # Generate the train and test data set for cross validation    
    #indexListSize = len(indexList)
    indexListSize = 0
    while(indexListSize != len(indexList)):
         # Shuffle the test list
         cvDic['TEST'].append(indexList[indexListSize])
         # Temporary train lsit
         trainList = list()
         for i in range(0,len(data['data'])):
             if (i not in indexList[indexListSize-1]):
                trainList.append(i)
         # Shuffle the train List
         random.shuffle(trainList)
         cvDic['TRAIN'].append(trainList)

         indexListSize += 1

    return cvDic         
                                                    
                                                                  
                                                                      
# Initialize all input layer to hidden layer weight
# Initialize all hidden layer to output layer weights
def _initialize_weights(data):

    # Define the input to hidden layer weight list
    # Define the hidden to output layer weight list
    _input_to_hidden_weight = list()
    _hidden_to_output_weight = list()
 
    for i in range(0, len(data['data'][0])-1):
        # Define the random value list from all input to one hidden unit
        _input_to_one_hidden_weight = list()
        for j in range(0, len(data['data'][0])):
            _input_to_one_hidden_weight.append(random.uniform(-0.01,0.01))
        _input_to_hidden_weight.append(_input_to_one_hidden_weight)

    for i in range(0,len(data['data'][0])):
        _hidden_to_output_weight.append(random.uniform(-0.01,0.01))

    return _input_to_hidden_weight, _hidden_to_output_weight




# Get the random initialized weight
# Calculate the output of the neural net
def _calculate_output(num_folds, data, rate, num_epochs):
    
    # Get the training and testing set
    cvDic = _n_fold_stratified_cross_validation(num_folds, data)
    trainSet = cvDic['TRAIN']
    testSet = cvDic['TEST']

    print("fold_of_instance", num_folds)

    # Define the accuracy list
    accuracyList = list()
 
    for trainIndex in range(0,len(trainSet)):
         # Get the random Initialized weight
         #_input_to_hidden_weight, _hidden_to_output_weight = _initialize_weights(data)
         # Iterate the trainSet
         counter = 0
         # Get the random Initialized weight
         _input_to_hidden_weight, _hidden_to_output_weight = _initialize_weights(data)

         while(counter != num_epochs):
       
             # Get the random Initialized weight
                 for instanceIndex in trainSet[trainIndex]:               
                  
                      # Iterate the certain data part line for input to hidden
                      # Define the output list 
                      o_hidden_List = list()
                      for i in range(0,len(data['data'][0])-1):
                          net = 0
                    
                          for j in range(0,len(data['data'][0])-1):
                              net += data['data'][instanceIndex][j]*_input_to_hidden_weight[i][j]
                          net+=_input_to_hidden_weight[i][-1] 
                     
                          # Output values for hidden unit
                          
                          o = 1/(1+math.exp(-net))
                          o_hidden_List.append(o)
                      o_hidden_List.append(1)
                 
                      # Calculate the output for hidden to output layer
                      # Initialize the output layer value
                      net_output = 0
                      for i in range(0,len(o_hidden_List)):
                          net_output += o_hidden_List[i]*_hidden_to_output_weight[i]
                      
                      output =  1/(1+math.exp(-net_output))
                      
                      # Epochs part to update the weight
                      # Encode the first class label to be 0
                      # Encode the second class label to be i1
                      if(data['data'][instanceIndex][-1] == data['attributes'][-1][1][0]): #actual = first class
                         output_df = output*(1-output)*(0-output)
                      else:    #actual = second class
                         output_df = output*(1-output)*(1-output)

                      for hIndex in range(0,len(_hidden_to_output_weight)):  
                          #print("Before", _hidden_to_output_weight[hIndex])     
                          _hidden_to_output_weight[hIndex] += rate*output_df*o_hidden_List[hIndex]
                          #print("delt weight ", rate*output_df*o_hidden_List[hIndex])
                          #print("Updated", _hidden_to_output_weight[hIndex])
                      # Define the parameter list
                      parameterList = list() #seta for hidden layer

                      for i in range(0,len(o_hidden_List)-1):
                          parameter = o_hidden_List[i]*(1-o_hidden_List[i])*output_df*_hidden_to_output_weight[i]
                          #print(parameter)
                          parameterList.append(parameter)            
                      
                      #print(_input_to_hidden_weight[0])  
                      for m in range(0,len(o_hidden_List)-1):
                          for n in range(0,len(data['data'][0])-1):
                              _input_to_hidden_weight[m][n] += rate*parameterList[m]*data['data'][instanceIndex][n] 
                              #print("rate ", rate)
                              #print("parameterList[m] ", parameterList[m])
                              #print("data['data'][instanceIndex][n] ", data['data'][instanceIndex][n])
                          _input_to_hidden_weight[m][-1] += rate*parameterList[m] 
                      #print(_input_to_hidden_weight[0])
                      #return
                 counter += 1                                
         
         # Define the correct counter
         correct = 0

         # Define the total correct prediction counter for each fold
         # Use the updated weight to test the test set accuracy
         for instanceIndex in testSet[trainIndex]:
             
             # Define the hidden output list for each instance
             test_o_hidden_List = list()
             for i in range(0,len(data['data'][0])-1):
                 # net value for each hidden unit
                 net = 0
                 for j in range(0,len(data['data'][0])-1):                  
                     net += data['data'][instanceIndex][j]*_input_to_hidden_weight[i][j]
                 net += _input_to_hidden_weight[i][-1]
 
                 # Sigmoid output value for each hidden unit                         
                 oHidden = 1/(1+math.exp(-net))
                 test_o_hidden_List.append(oHidden)
             test_o_hidden_List.append(1)
            
             # Define the output net value
             # Calculate the output value
             output_net = 0

             for i in range(0,len(test_o_hidden_List)):
                 output_net += test_o_hidden_List[i]*_hidden_to_output_weight[i]
              
             output = 1/(1+math.exp(-output_net))

             if(output < 0.5 and data['data'][instanceIndex][-1] == data['attributes'][-1][1][1]):
                print("predicted_class ", data['attributes'][-1][1][0], " actual_class ", data['attributes'][-1][1][1])                            
 
             if(output >= 0.5 and data['data'][instanceIndex][-1] == data['attributes'][-1][1][0]):
                print("predicted_class ", data['attributes'][-1][1][1], " actual_class ", data['attributes'][-1][1][0])
       
             if(output < 0.5 and data['data'][instanceIndex][-1] == data['attributes'][-1][1][0]):
                print("predicted_class ", data['attributes'][-1][1][0], " actual_class ", data['attributes'][-1][1][0])
                correct += 1 
         
             if(output >= 0.5 and data['data'][instanceIndex][-1] == data['attributes'][-1][1][1]):
                print("predicted_class ", data['attributes'][-1][1][1], " actual_class ", data['attributes'][-1][1][1])
                correct += 1

         accuracy = correct/len(testSet[trainIndex])
         accuracyList.append(accuracy)                         
    
    # Calculate the total accuracy
    sum = 0
    for item in accuracyList:
        sum += item             
    print("confidence_of_prediction ", sum/len(accuracyList))
   


 



# Open the arff file as the second argument
# Number of folds as the third argument
# The learning rate as the fourth argument
# The number of epochs as the fifth argument
data = arff.load(open(sys.argv[1], 'r'))
numFolds = int(sys.argv[2])
learningRate = float(sys.argv[3])
numEpochs = int(sys.argv[4])




_calculate_output(numFolds, data, learningRate, numEpochs)
