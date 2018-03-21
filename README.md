# Frid.AI
![Robots](https://mmtstock.com/wp-content/uploads/2014/09/PB_20140912201212155.jpg)

## What is this?
Minimal examples for basic artificial neural network architectures,
problems, layers and general techniques.

These examples are targeted at coders without experience in using
artificial neural networks.

I will try and expand the number of examples to cover a wide spectrum
of different areas and use cases.

## How to install?
1. Clone this repository.
2. Download python (https://www.python.org/downloads/) if you dont have
it already (careful: 64bit is needed)
3. Be sure `pip` as well as `python` are in your `PATH` 
4. Navigate to where you cloned this repository and enter the folder
FridAI
5. Use `pip install -r ./requirements.txt` to install all required
packages
6. Check everything is working by running a arbitrary example
(`python -m digits`) It should load a few seconds, after which it will
print the neural network's training progress.
7. Wait for the process to finish without errors

## What to do now?
You already ran the digits example, while the network trained, it
produced logs, which can be plotted.

Run `tensorboard --logdir=digits/logs --host=localhost` and use your
favourite browser to navigate to https://localhost:6006, where you can
view the training progress. For more infos check
[this section](##how-do-i-read-the-training-plots)!

After you correctly set up everything, have a look at the
`digits/solution.py`. It contains a working example of a simple yet
interesting problem solved with an artificial neural network.

The (more simple) artificial neural network is created in
`digits/models/mlp.py`. Performance of this model is heavily dependent
on so called hyper-parameters, which are for example the units of a
layer, the number of hidden layers, activation function in hidden/output
layer and many more. Try and experiment with the value of these a little!

## Where can I get help with Keras?
Keras is very well documented, you can find it here: https://keras.io/

Layers are documented here: https://keras.io/layers/core/

Activation functions are here: https://keras.io/activations/

The model (fit/predict functions) is documented here:
https://keras.io/models/model/

### Ok got it, where can I find more advanced tips?
Once you played around with different layer hyper parameters, you can
get your feet wet with alternative loss functions or optimizers.

Loss functions you can find here: https://keras.io/losses/

Optimizers are documented here: https://keras.io/optimizers/

### I have done all your examples, what now?
There are plenty of use cases, if you feel like you want to practice
with prepared datasets a little bit more though, check here:
https://keras.io/datasets/

## How do I read the training plots?
There are (most commonly) four different plots:
- *loss*: error the network produces while __training__
- *accuracy*: accuracy in predicting the correct label/class while
 __training__
- *__val__-loss*: error the network produces while __testing__ it,
which means: error it produces while presenting samples, the network
has never seen
- *__val__-accuracy*: accuracy in predicting the correct label/class
while __testing__ it, which means: accuracy in predicting the correct
label/class while presenting samples, the network has never seen

The x-axis of each plot is the training progress, measured in epochs.
One epoch is done, when the entirety of training data was presented to
the neural net once.

The y-axis is the loss(error)/accuracy (or whatever metric is measured).
Low loss and high accuracy are desirable.