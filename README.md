# Frame-Interpolation
> Motion Compensated Frame Interpolation using Deep ConvoNets

## About

Frame Inerpolation is the technique of generating intermediate frames between existing ones, by the method the interpotion, such the resulting frame sequence may be smoother and more continuous.
Interpolation can be achieved by a number of methods, averaging being the simplest of the lot. This repository tries to illustrate the process of interpolation by making use of a deep convolutional neural networks.


## Usage

First of all, install all the required libraries with the help of the following command

    pip3 install -r requirements.txt

To explore some code, run any of the notebooks in the `Clips`, `Model` or `Dataset` directories. 
Next, to see some examples of interpolation, view any pair of corresponding clips present within the `Examples` directory.

To interpolate any of your own clips, run the following command

    python3 fps.py [filepath] -u [interpolation factor]
    
where *filepath* is the path to the clip you want to interpolate, and *interpolation factor* is an integer multiplier

You can also downscale the frame rate of any of your clips by using the command

    python3 fps.py [filepath] -d [reduction factor]
