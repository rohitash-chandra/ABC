# Basic Approximate Bayseian Computation
# Rohitash Chandra and Sally Cripps (2017).
# CTDS, UniSYD. c.rohitash@gmail.com
# Simulated data is used.


# Ref: https://en.wikipedia.org/wiki/Dirichlet_process
# https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.random.dirichlet.html

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import matplotlib.mlab as mlab
import time

def model(nModels, x, mu, sig, w):
    fx = np.zeros(x.size)
    for i in range(nModels):
        fx = fx + w[i] * mlab.normpdf(x,mu[i],np.sqrt(sig[i]))   # Weighted Mixture of Normal Distribution
    return fx




def sampler(samples, nModels, x, ydata):
    portionaccept = int(samples * 0.001) # an estimate of max accepted
    #create space to store accepted samples for posterior
    pos_mu = np.ones((portionaccept , nModels))
    pos_sig = np.ones((portionaccept , (nModels)))
    pos_w = np.ones((portionaccept , (nModels)))
    pos_tau = np.ones(portionaccept )
    pos_fx = np.ones((portionaccept , ydata.size))

    #create space to store fx of all samples
    fx_samples = np.ones((samples, ydata.size))
    # ygen is the fx of every sample after noise is added
    ygen = np.ones((samples, ydata.size))
    diff = np.ones((samples, ydata.size))
    mse= np.zeros(samples) # space for Mean Squared Error

    upperlim_sig = 2 * np.var(x) # way to create new proposals based on data
    upperlim_tau = 1 * np.var(ydata)


    epsilon = 0.25 # defines how you accept proposals

   #draw samples for mean, sigma and weights, tau from uniform distribution
    sig_current  =  np.random.rand(samples,nModels) * upperlim_sig
    mu_current =  np.random.rand(samples,nModels)
    w_current  =  np.random.dirichlet((1,1), samples) # will work for 2 Normal distributions
    tau_current = np.random.rand(samples,1) * upperlim_tau

    k = 0
    for h in range(samples):
       fx_samples[h,: ]=  model(nModels, x, mu_current[h,:], sig_current[h,:], w_current[h,:]) #evaluate proposal through model
       ygen[h,: ] =  fx_samples[h,: ] + np.random.normal(0, tau_current[h], ydata.size) # noise from Normal distribution
       diff[h,: ] = np.square(ygen[h,: ] - ydata)
       mse[h] =  np.sqrt( np.sum(diff[h,:])/x.size) # get mean squred error

       if mse[h] < epsilon: # define how you accept 
          pos_mu[k,:] = mu_current[h,:]
          pos_sig[k,:] = sig_current[h,:]
          pos_w[k,:] = w_current[h,:]
          pos_tau[k] = tau_current[h]
          pos_fx[k,:] = fx_samples[h,: ]
          k = k+1

    pos_fx_ = pos_fx[0:int(k),]
    mse_ = mse[0:int(k)]

    print k,  'is number accepted'


    fx = fx_samples[0,:]
    fx_begin = ygen[0,:]

    plt.plot(x, ydata)
    plt.plot(x, fx )
    plt.plot(x, fx_begin )

    plt.title("Plot of Data vs Initial Fx")
    plt.savefig('results/begin.png')
    plt.clf()


    fx_mu = pos_fx_.mean(axis=0)
    fx_high = np.percentile(pos_fx_, 95, axis=0)
    fx_low = np.percentile(pos_fx_, 5, axis=0)

    plt.plot(x, ydata)
    plt.plot(x, fx_mu)

    plt.plot(x, fx_low)
    plt.plot(x, fx_high)
    plt.fill_between(x, fx_low, fx_high,facecolor='g',alpha=0.4)

    plt.title("Plot of Data vs  Uncertainty ")
    plt.savefig('results/abcres.png')
    plt.clf()

    return (mse_, pos_fx_, k)


def main():


	random.seed(time.time())
        nModels = 2

        modeldata= np.loadtxt('simdata.txt') # load univariate data in same format as given
        #print modeldata

        ydata = modeldata  #
        print ydata.size
        x = np.linspace(1/ydata.size, 1, num=ydata.size) #   (  input x for ydata)

        NumSamples = 2000000   # need to pick yourself

	[mse, pos_fx, k] = sampler(NumSamples, nModels, x, ydata)
    	print 'mse and posterior fx of accepted samples'
    	print mse
    	print pos_fx





if __name__ == "__main__": main()
