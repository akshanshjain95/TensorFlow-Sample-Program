import tensorflow as tf
import sys


# Adding one to W
# ~~~
def addOne():
    sess.run(tf.assign(W, sess.run(W) + 1.0))


# ASubtracting one from W
# ~~~
def subtractOne():
    sess.run(tf.assign(W, sess.run(W) - 1.0))


# Defining model parameters
# ~~~
W = tf.Variable(1, dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Defining model and loss function
# ~~~
myModel = W * x
delta = myModel - y
squaredDelta = tf.square(delta)
loss = tf.reduce_sum(squaredDelta)

# Initializing oldLoss to maximum float value
# ~~~
oldLoss = sys.float_info.max

# Initializing global variables
# ~~~
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Initializing flags to remember what operation was last executed
# ~~~
adding = 0
subtracting = 0

# Loop to determine the value of W / Model training
# ~~~
while oldLoss > 0:
    currentLoss = sess.run(loss, {x: [1, 2, 3, 4], y: [10, 20, 30, 40]})
    if currentLoss == 0:
        break
    elif adding == 0 and subtracting == 0:
        addOne()
        adding = 1
    elif adding == 1 and currentLoss <= oldLoss:
        addOne()
        adding = 1
        subtracting = 0
    elif adding == 1 and currentLoss >= oldLoss:
        subtractOne()
        adding = 0
        subtracting = 1
    elif subtracting == 1 and currentLoss <= oldLoss:
        subtractOne()
        subtracting = 1
        adding = 0
    elif subtracting == 1 and currentLoss >= oldLoss:
        addOne()
        subtracting = 0
        adding = 1
    oldLoss = currentLoss

# Printing the value of W
# ~~~
print("After training the value of W is -", sess.run(W))

# Printing the output for some inputs to see if the model is giving correct output
# ~~~
print("--------------------------------------------------")
print("Should return 10 times of 27 -", sess.run(myModel, {x: 27.0}))
print("Should return 10 times of 10 -", sess.run(myModel, {x: 10.0}))
print("Should return 10 times of 80 -", sess.run(myModel, {x: 80.0}))
