# MachineLearning_HW5
Implement VGG-16 network architecture 

# Part 1:
Implement VGG-16 network architecture in Keras or use the existing
implementation and apply it to MNIST database. i) Train your network from scratch
(randomly initialize your network) with enough epochs till the training error is
saturated. ii) use a VGG-16 network that is trained on ImageNet and retrain it on MNIST. Report the accuracy and
training error on both the validation and training set. You need to resize your images,
so they can be passed to the VGG16 network. Replicate the grayscale channel to
accommodate for the RGB nature of the input Conv Layer. You also need to adjust
the last layer of the VGG network to accommodate the MNIST 10 class labels.

# Part 2:
Freeze all the Conv layers of the VGG-16 network trained on ImageNet and train the
network on MNIST. Report your results (accuracy and training error) and compare
them with the results of part 1.

# Part 3: 
Freeze all the FC layers of the VGG-16 network pre-trained on ImageNet and train
the network on MNIST. Report your results (accuracy and training error) and
compare them with the results of parts 1 & 2
