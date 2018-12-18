
import numpy as np
import matplotlib.pyplot as plt
# data I/O
filename = 'dataset.txt'
file = open(filename, 'r')
data = file.read()
# use set() to count the vacab size
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (data_size, vocab_size))

# dictionary to convert char to idx, idx to char
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 50 # size of hidden layer of neurons
seq_length = 50 # number of steps to unroll the RNN for
learning_rate = 1e-1
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias
def lossFun(inputs, targets, hprev):
  
  xs, hs, ys, ps = {}, {}, {}, {}
  ## record each hidden state of
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass for each training data point
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size, 1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    
    ## hidden state, using previous hidden state hs[t-1]
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
    ## unnormalized log probabilities for next chars
    ys[t] = np.dot(Why, hs[t]) + by
    ## probabilities for next chars, softmax
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
    ## softmax (cross-entropy loss)
    loss += -np.log(ps[t][targets[t], 0])

  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    ## compute derivative of error w.r.t the output probabilites
    ## dE/dy[j] = y[j] - t[j]
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y
    
    ## output layer doesnot use activation function, so no need to compute the derivative of error with regard to the net input
    ## of output layer. 
    ## then, we could directly compute the derivative of error with regard to the weight between hidden layer and output layer.
    ## dE/dy[j]*dy[j]/dWhy[j,k] = dE/dy[j] * h[k]
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    
    ## backprop into h
    ## derivative of error with regard to the output of hidden layer
    ## derivative of H, come from output layer y and also come from H(t+1), the next time H
    dh = np.dot(Why.T, dy) + dhnext
    ## backprop through tanh nonlinearity
    ## derivative of error with regard to the input of hidden layer
    ## dtanh(x)/dx = 1 - tanh(x) * tanh(x)
    dhraw = (1 - hs[t] * hs[t]) * dh
    dbh += dhraw
    
    ## derivative of the error with regard to the weight between input layer and hidden layer
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    ## derivative of the error with regard to H(t+1)
    ## or derivative of the error of H(t-1) with regard to H(t)
    dhnext = np.dot(Whh.T, dhraw)

  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

## given a hidden RNN state, and a input char id, predict the coming n chars
def sample(h, seed_ix, n):
  
  ## a one-hot vector
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1

  ixes = []
  for t in range(n):
    ## self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    ## y = np.dot(self.W_hy, self.h)
    y = np.dot(Why, h) + by
    ## softmax
    p = np.exp(y) / np.sum(np.exp(y))
    ## sample according to probability distribution
    ix = np.random.choice(range(vocab_size), p=p.ravel())

    ## update input x
    ## use the new sampled result as last input, then predict next char again.
    x = np.zeros((vocab_size, 1))
    x[ix] = 1

    ixes.append(ix)

  return ixes


## iterator counter
n = 0
## data pointer
p = 0

mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
l=[]
sl=[]
## main loop
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p + seq_length + 1 >= len(data) or n == 0:
    # reset RNN memory
    ## hprev is the hiddden state of RNN
    hprev = np.zeros((hidden_size, 1))
    # go from start of data
    p = 0

  inputs = [char_to_ix[ch] for ch in data[p : p + seq_length]]
  targets = [char_to_ix[ch] for ch in data[p + 1 : p + seq_length + 1]]

  # sample from the model now and then
  if n % 1000 == 0:
    sample_ix = sample(hprev, inputs[0], 5000)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print ('---- sample -----')
    print ('----\n %s \n----' % (txt, ))
    
  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  ## author using Adagrad(a kind of gradient descent)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 1000 == 0:
    print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress
    l.append(n)
    sl.append(smooth_loss)             
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) 

  p += seq_length # move data pointer
  n += 1 # iteration counter 
  
# gradient checking
  
from random import uniform
def gradCheck(inputs, target, hprev):
  global Wxh, Whh, Why, bh, by
  num_checks, delta = 10, 1e-5
  _, dWxh, dWhh, dWhy, dbh, dby, _ = lossFun(inputs, targets, hprev)
  for param,dparam,name in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby], ['Wxh', 'Whh', 'Why', 'bh', 'by']):
    s0 = dparam.shape
    s1 = param.shape
    if(s0 == s1): 
        print('Error dims dont match: %s and %s.' % (s0, s1))
    print (name)
    for i in range(num_checks):
      ri = int(uniform(0,param.size))
      
      old_val = param.flat[ri]
      param.flat[ri] = old_val + delta
      cg0, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
      param.flat[ri] = old_val - delta
      cg1, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
      param.flat[ri] = old_val 
      grad_analytic = dparam.flat[ri]
      grad_numerical = (cg0 - cg1) / ( 2 * delta )
      rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
      print ('%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
     
gradCheck(inputs,targets,hprev)

      
plt.plot(loss,smoothloss,label='Epoch Loss')

plt.xlabel('Epochs') 

plt.ylabel('Loss') 

plt.title('EPOCH _LOSS PLOT') 
  

plt.legend() 
  
plt.show()