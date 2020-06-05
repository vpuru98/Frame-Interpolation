
# Model

The model that I am currently using is a 13 layer deep convolutional neural network. All the layers of the network use the `relu` activation function. The model does not contain any down-sampling layers. A summary of the model is presented below :
    
    Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 80, 80, 6)         0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 78, 78, 200)       11000     
    _________________________________________________________________
    activation_1 (Activation)    (None, 78, 78, 200)       0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 76, 76, 200)       360200    
    _________________________________________________________________
    activation_2 (Activation)    (None, 76, 76, 200)       0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 76, 76, 200)       360200    
    _________________________________________________________________
    activation_3 (Activation)    (None, 76, 76, 200)       0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 76, 76, 250)       450250    
    _________________________________________________________________
    activation_4 (Activation)    (None, 76, 76, 250)       0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 76, 76, 250)       562750    
    _________________________________________________________________
    activation_5 (Activation)    (None, 76, 76, 250)       0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 76, 76, 250)       562750    
    _________________________________________________________________
    activation_6 (Activation)    (None, 76, 76, 250)       0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 76, 76, 300)       675300    
    _________________________________________________________________
    activation_7 (Activation)    (None, 76, 76, 300)       0         
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 76, 76, 300)       810300    
    _________________________________________________________________
    activation_8 (Activation)    (None, 76, 76, 300)       0         
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 76, 76, 300)       810300    
    _________________________________________________________________
    activation_9 (Activation)    (None, 76, 76, 300)       0         
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, 76, 76, 350)       945350    
    _________________________________________________________________
    activation_10 (Activation)   (None, 76, 76, 350)       0         
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 76, 76, 350)       1102850   
    _________________________________________________________________
    activation_11 (Activation)   (None, 76, 76, 350)       0         
    _________________________________________________________________
    conv2d_12 (Conv2D)           (None, 76, 76, 350)       1102850   
    _________________________________________________________________
    activation_12 (Activation)   (None, 76, 76, 350)       0         
    _________________________________________________________________
    conv2d_13 (Conv2D)           (None, 76, 76, 3)         51453     
    _________________________________________________________________
    activation_13 (Activation)   (None, 76, 76, 3)         0         
    =================================================================
    Total params: 7,805,553
    Trainable params: 7,805,553
    Non-trainable params: 0
    _________________________________________________________________ 


The cost function that this model is trained against consists of two parts. The first part is the plain old mean-squared error function. The second part is a custom function which tries to capture the degree of definition of edges in any image, and penalizes the model for producing images that differ in edge specificity from the desired image. 

**PS.** I also tried using residual nets for training, but that did not seem to offer any appreciable improvement over the current model.

