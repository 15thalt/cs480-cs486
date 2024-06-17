import numpy as np
from math import comb

def Perceptron(X, y, w=None, b=0, max_pass=500):
    # dimension
    d = X.shape[0]
    # datapoints
    n = X.shape[1]

    # set w if not given
    if w is None:
        w = np.zeros(d)

    # create empty array of mistakes per pass-through
    mistakes = np.zeros(max_pass)

    # run through dataset [max_pass] times
    for t in range(max_pass):
        mistakes[t] = 0

        # run through each datapoint
        for i in range(n):

            # if mistake made, update accordingly
            if y[i] * (np.dot(X[:, i], w) + b) <= 0:
                w = w + y[i] * X[:, i]
                b = b + y[i]
                mistakes[t] += 1
    
    # return found parameters, mistake count
    return w, b, mistakes



# only difference is each row is an entry instead of column in Perceptron()
def PerceptronFlip(X, y, w=None, b=0, max_pass=500):
    # dimension
    d = X.shape[1]
    # datapoints
    n = X.shape[0]

    # set w if not given
    if w is None:
        w = np.zeros(d)

    # create empty array of mistakes per pass-through
    mistakes = np.zeros(max_pass)

    # run through dataset [max_pass] times
    for t in range(max_pass):
        mistakes[t] = 0

        # run through each datapoint
        for i in range(n):

            # if mistake made, update accordingly
            if y[i] * (np.dot(X[i], w) + b) <= 0:
                w = w + y[i] * X[i]
                b = b + y[i]
                mistakes[t] += 1
    
    # return found parameters, mistake count
    return w, b, mistakes



def Perceptron1vA(X, y, labels, w=None, b=0, max_pass=500):
    # datapoints
    n = X.shape[0]
    # dimension
    d = X.shape[1]

    # init W-array to hold (#labels) w's (where w=[b,w])
    W = np.zeros((len(labels),d+1))
    

    # for each label
    for k in range(len(labels)):

        # build indicator label (true/false)
        ind = np.zeros(n)
        for i in range(n):
            if y[i] == labels[k]:
                ind[i] = 1
            else:
                ind[i] = -1
        
        # run perceptron on ind
        w_tmp, b_tmp, twy = PerceptronFlip(X, ind, w, b, max_pass)

        # store accordingly
        W[k,0] = b_tmp
        W[k, 1:d+1] = w_tmp

    # return W-array of (#labels) w's
    return W



def Perceptron1v1(X, y, labels, w=None, b=0, max_pass=500):
    # datapoints
    n = X.shape[0]
    # dimension
    d = X.shape[1]
    # length of labels
    ll = len(labels)

    # init W-matrix of (llxll)-size (only using upper-triangle) of w's 
    W = np.zeros((ll,ll,d+1))

    # for each combination
    for i in range(ll):
        for j in range(i+1,ll):

            # filter relevant features/labels
            label1 = labels[i]
            label2 = labels[j]
            indexes = np.where((y == label1) | (y == label2))
            y_pair = y[indexes]
            X_pair = X[indexes]

            # create indicators
            ind = np.zeros(len(y_pair))
            for a in range(len(y_pair)):
                if y_pair[a] == label1:
                    ind[a] = 1
                elif y_pair[a] == label2:
                    ind[a] = -1
            
            # run perceptron on relevant ind, relevant features
            w_tmp, b_tmp, twy = PerceptronFlip(X_pair, ind, w, b, max_pass)

            # store accordingly
            W[i,j,0] = b_tmp
            W[i,j,1:d+1] = w_tmp
    
    # return W-array of (#labelsC2) w's
    return W



def PerceptronAvA(X, y, labels, w=None, b=0, max_pass=500):
    # datapoints
    n = X.shape[0]
    # dimension
    d = X.shape[1]
    # length of labels
    ll = len(labels)

    # alter X = [1, X]
    X_new = np.hstack((np.ones((n,1)), X))

    # set w if not given
    if w is None:
        w = np.zeros(d)

    # create empty array of mistakes per pass-through
    mistakes = np.zeros(max_pass)

    # init W-array to hold (#labels) w's (where w=[b,w])
    W = np.zeros((len(labels),d+1))

    # run through dataset [max_pass] times
    for t in range(max_pass):
        mistakes[t] = 0

        # run through each datapoint
        for i in range(n):
            made_mistake = False

            # get true label and its index
            y_i = y[i]
            y_idx = labels.index(y_i)

            # find relevant dp
            y_dp = np.dot(X_new[i], W[y_idx])


            # for each label
            for l_idx in range(ll):

                # get corresponding w
                w_focus = W[l_idx]

                # if same label
                if l_idx == y_idx: continue

                # compute dot product
                dp = np.dot(X_new[i], w_focus)

                # compare against y_dp
                # if 'dominating', update accordingly
                if dp - y_dp >= 0:
                    made_mistake = True
                    W[l_idx] -= X_new[i] # want dot product to decrease
                    W[y_idx] += X_new[i] # want dot product to increase (tech. this can update up to (ll - 1)-times)
                    y_dp = np.dot(X_new[i], W[y_idx]) # update relevant dp

            # increment if made a mistake
            if made_mistake: mistakes[t] += 1
        # print("on iteration " + str(t) + ": made " + str(mistakes[t]) + " mistakes")
                    
    
    # return W-array of (#labels) w's
    return W, mistakes


def predictPerceptron1vA(X, W, labels):
    # datapoints
    n = X.shape[0]
    # alter X = [1, X]
    X_new = np.hstack((np.ones((n,1)), X))
    # init for n predictions
    y_pred = np.zeros(n)

    # for-each datapoint
    for i in range(n):
        max_dp = None
        max_label = None

        # for-each label
        for j in range(len(labels)):

            # calc. <datapoint, w-label>
            dp = np.dot(X_new[i], W[j])

            # if better, update accordingly
            if max_dp == None or max_dp < dp:
                max_label = labels[j]
                max_dp = dp

        # store best guess
        y_pred[i] = max_label

    # return n-guesses
    return y_pred



def predictPerceptron1v1(X, W, labels):
    # datapoints
    n = X.shape[0]
    # alter X = [1, X]
    X_new = np.hstack((np.ones((n,1)), X))
    # length of labels                          
    ll = len(labels)
    # init for prediction                                                        
    Y_hat = np.zeros(n)

    # for-each datapoint                                      
    for idx, x in enumerate(X_new):                                                           
        y_hat = None
        max_sum = None

        # for each label
        for k in range(ll):
            sum = 0

            # check against all label pairs
            for l in range(ll):

                # find appropriate <label, label>-w in W
                flip = False
                if l == k: continue
                if (l < k):
                    flip = True
                    w = W[l,k]
                else: w = W[k,l]

                # find dot product and add appropriate ind to sum
                dp = np.dot(x,w)
                ind = None
                if (dp > 0):
                    ind = 1
                elif (dp <= 0):
                    ind = 0
                if (flip): 
                    if ind == 1: 
                        ind = 0
                    elif ind == 0:
                        ind = 1
                sum += ind
            
            # init or update if better guess found
            if y_hat is None or max_sum < sum:
                y_hat = labels[k]
                max_sum = sum
        
        # store best-guess
        Y_hat[idx] = y_hat
    
    # return n best-guesses
    return Y_hat          
      
            
def predictPerceptronAvA(X, W, labels):
    # datapoints
    n = X.shape[0]
    # alter X = [1, X]
    X_new = np.hstack((np.ones((n,1)), X))
    # init for n predictions
    y_pred = np.zeros(n)

    # for-each datapoint
    for i in range(n):
        max_dp = None
        max_label = None

        # for-each label
        for j in range(len(labels)):

            # calc. <datapoint, w-label>
            dp = np.dot(X_new[i], W[j])

            # if better, update accordingly
            if max_dp == None or max_dp < dp:
                max_label = labels[j]
                max_dp = dp

        # store best guess
        y_pred[i] = max_label

    # return n-guesses
    return y_pred   

# compare predictions versus true values
def error(y_pred, y_true):
    mistakes = 0
    for i in range(y_true.size):
        if y_true[i] != y_pred[i]:
            mistakes += 1
    return mistakes, y_true.size

