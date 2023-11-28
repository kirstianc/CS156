# Rosenblatt Perceptron Algorithm

import numpy as np

def sgn(n, b):
    if round(n - b) > 0: # Assumes Q is always This class 0?
        print (False)
        return False 
    elif round(n - b) <= 0:
        print (True)
        return True
    print ("Unknown")
    return None

X = np.matrix([[0,0], [1,1], [0,1], [1,0]])
#X_test = np.matrix([[0, 0.5]])
X_test = np.matrix([[0, .1]])
W = np.matrix([[0, 0], [0, 0], [0, 0], [0, 0]])
W_orig = W
b = np.matrix([[0], [0], [0], [0]])
b_orig = b
y = np.matrix([[0], [1], [0], [1]])
y_orig = y
a = 0.1

print ("Calculate R^2") 
Rsqd = np.max(np.apply_along_axis(np.linalg.norm, axis=1, arr=X))**2.
print ("R squared = ", Rsqd)
print

for i in range(10):
    print (">>>>>>>>>>>>>>>>>>>>>>>>>")
    print ("     ITERATION ", i+1)
    print (">>>>>>>>>>>>>>>>>>>>>>>>>")
    print ("W = ", W)
    wx = W.dot(X.T)
    print ("W.dot(X.T) = ", wx)
    wxs = wx.sum(axis=1)
    print
    print ("wxs = ", wxs)
    wxsb = wxs + b
    print
    print ("wsxb = ", wxsb)
    print
    
    print ("iteration error: 1/s SUM(|d-yi(t)|)")
    iterr = (np.absolute((y - wxsb)).sum()) / float(len(W))
    print ("iteration error = ", iterr)
    print ()
    
    print ("a = ", a)
    aywxs = a*(y - wxsb)
    print ("a*(y-wxsb) =", aywxs)
    print()
    Waywxsb = (a*(y-wxsb))+W
    print ("(a*(y-wxsb))+W = ", Waywxsb)
    print()
    print ("Updating original W matrix.")
    W = Waywxsb
    print ("W = ", W)
    print()
    
    print ("Updating b matrix" )
    b = b + (a * (y - wxsb)) * Rsqd
    print (" b = ", b)
    
print
print (">>>>>>>>>>>>>>>")
print ("Now testing X_test")
wxtest = W.dot(X_test.T)
print ("W.dot(X_test) = ", wxtest)
wxtests = wxtest.sum(axis=1)
print()
print ("wxtests = ", wxtests)
wxtestsb = wxtests + b
print
print ("wsxtestb = ", wxtestsb)
print
r = wxtestsb.sum()
print ("wxtestsb.sum() = ", r)
print ("Is this class 0: ", sgn(r, b.sum()), " for X_test = ", X_test)
print ()
#print ("sum of b = ", b.sum()
print
print ("Size of ||w|| = ", np.max(np.apply_along_axis(np.linalg.norm, axis=1, arr=W))**2.)