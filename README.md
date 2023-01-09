# Natural Language Processing with Pytorch | KFold | Optuna



### PyTorch 
  - `Pytorch` is a machine learning framework based on the Torch library, used for applications such as computer vision and natural language processing, originally developed by Meta AI(facebook).Although the Python interface is more polished and the primary focus of development, PyTorch also has a C++ interface.


### K fold Cross Validation
  - `K fold` Cross Validation is a technique used to evaluate the performance of your machine learning or deep learning model in a robust way.
  - It splits the dataset into k parts/folds of approximately equal size. Each fold is chosen in turn for testing and the remaining parts for training.
  - This process is repeated k times and then the performance is measured as the mean across all the test sets.
  
### Optuna | A hyperparameter optimization framework 
  - To learn more information related `optuna` follow link https://optuna.org/
  
### F1 Score 


    F1 is calculated as follows:
    𝐹1= $2*Precision*Recall$
    where:

    𝑝𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛=𝑇𝑃𝑇𝑃+𝐹𝑃
    𝑟𝑒𝑐𝑎𝑙𝑙=𝑇𝑃𝑇𝑃+𝐹𝑁
    and:

    True Positive [TP] = your prediction is 1, and the ground truth is also 1 - you predicted a positive and that's true!
    False Positive [FP] = your prediction is 1, and the ground truth is 0 - you predicted a positive, and that's false.
    False Negative [FN] = your prediction is 0, and the ground truth is 1 - you predicted a negative, and that's false.
  
  
## Preprocessing steps 

 
  
  
  
## Implementation with Pytorch and sklearn

* The K Fold Cross Validation is used to evaluate the performance of the RNN(LSTM) model on the dataset. This method is implemented using the sklearn    library, while the model is trained using Pytorch.Happening all of this we tuned parameters with `optuna`.

* We define the  Neural network architecture with one  LSTM layers and one fully connected layer to classify the sentences into one of the two categories. We add two Dropout layers in the model to limit the risk of overfitting.

* We initialize the nn.BCEWithLogitsLoss loss function for the classification, the device to utilize the GPU.

* Moreover, we generate 4 folds using the Kfold function in each trails of objective function , where we have random splits and replicable results with random_state=0 . So, it divides the dataset into 3 parts for training and the remaining part for testing.
