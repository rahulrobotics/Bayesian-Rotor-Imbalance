import numpy as np
import pymc3 as pm
import theano.tensor as tt
import warnings
from scipy.optimize import approx_fprime

#-------------------------------------------return imbalance frequency response-----------------------------------------
def getImbalanceResponse(rotor_model, obs_node, magnitude, phase, frequency_range):
    '''
    rotor_model: the rotor object defined using ROSS library
    obs_node: node at which imbalance force is to be applied
    magnitude: magnitude of imbalance
    phase: phase of imbalance (Radians)
    frequency_range: range of frequencies which are being considered
    return:
        results: a results object
        fr: a matrix of the magnitudes and phases in the x and y directions
    '''
    probex = (1, 0)
    probey = (1, 90)

    results = rotor_model.run_unbalance_response(obs_node, magnitude, phase, frequency_range)
    x_mag = results.data_magnitude(probe=[probex], probe_units="degrees").loc[:, 'Probe 1 - Node 1'].values
    x_phase = results.data_phase(probe=[probex], probe_units="degrees").loc[:, 'Probe 1 - Node 1'].values
    y_mag = results.data_magnitude(probe=[probey], probe_units="degrees").loc[:, 'Probe 1 - Node 1'].values
    y_phase = results.data_phase(probe=[probey], probe_units="degrees").loc[:, 'Probe 1 - Node 1'].values
    fr = np.vstack((x_mag, x_phase, y_mag, y_phase))

    return results, fr

#---------------------------------------return gaussian likelihood------------------------------------------------------
# function needs validation

def my_loglike(theta, fr_obs, sigma, rotor_model, obs_node, frequency_range):
    """
    A Gaussian log-likelihood function for a model with parameters given in theta

    inputs:
        theta: parameters
        fr_obs: observed frequency response
        sigma: standard deviation of gaussian likelihood
        rotor_model: the rotor object defined using ROSS library
        obs_node: node at which imbalance force is to be applied
        frequency_range: range of frequencies which are being considered
    return:
        the logarithm of the normal likelihood of the data given the frequency response and the standard deviation
    """
    mag, phase = theta

    _, fr = getImbalanceResponse(rotor_model, obs_node, mag, phase, frequency_range)

    # # validation of arithmetic (
    # test_obs = [[1, 2], [3, 4]]
    # test_model = [[5, 6], [7, 8]]
    #
    # print('test: ', -(0.5 / 1 ** 2) * np.sum(np.sum(np.square(np.subtract(test_obs, test_model)), axis=1)))

    return -(0.5/sigma**2)*np.sum(np.sum(np.square(np.subtract(fr_obs, fr)), axis=1))

# numpy

#---------------------------------------return gradient of imbalance frequency response---------------------------------
def gradients(vals, func, releps=1e-3, abseps=None, mineps=1e-9, reltol=1e-3,
              epsscale=0.5):
    """
    Parameters
    ----------
    vals: array_like
        A set of values, that are passed to a function, at which to calculate
        the gradient of that function
    func:
        A function that takes in an array of values.

    Returns
    -------
    grads: array_like
        An array of gradients for each non-fixed value.
    """
    grads = approx_fprime(vals, func, epsilon = 1.49e-08)
    return grads
#-----------------------------------------theano op for rotor model-----------------------------------------------------
class RotorModelOp(tt.Op):

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dmatrix]  # outputs a single scalar value (the log likelihood)

    def __init__(self, get_imb_resp,rotor_model, obs_node, frequency_range):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        """
        # add inputs as class attributes
        self.get_imb_resp = get_imb_resp
        self.rotor_model = rotor_model
        self.obs_node = obs_node
        self.frequency_range = frequency_range

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (Beta,) = inputs  # this will contain my variables

        mag = Beta[0]
        ph = Beta[1]
        # call the log-likelihood function

        _, outputs[0][0] = getImbalanceResponse(self.rotor_model, self.obs_node, mag, ph, self.frequency_range)


#-----------------------------------------theano op for log likelihood response-----------------------------------------
class LogLikeWithGrad(tt.Op):

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, fr_obs, sigma, rotor_model, obs_node, frequency_range):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.fr_obs = fr_obs
        self.sigma = sigma
        self.rotor_model = rotor_model
        self.obs_node = obs_node
        self.frequency_range = frequency_range

        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.likelihood, self.fr_obs, self.sigma, self.rotor_model, self.obs_node, self.frequency_range)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.fr_obs, self.sigma, self.rotor_model, self.obs_node, self.frequency_range)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters
        return [g[0] * self.logpgrad(theta)]


#--------------------------------------------theano op for gradient of log likelihood response-------------------------------
class LogLikeGrad(tt.Op):
    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """

    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, loglike, fr_obs, sigma, rotor_model, obs_node, frequency_range):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """
        # add inputs as class attributes
        self.likelihood = loglike
        self.fr_obs = fr_obs
        self.sigma = sigma
        self.rotor_model = rotor_model
        self.obs_node = obs_node
        self.frequency_range = frequency_range

    def perform(self, node, inputs, outputs):
        (theta,) = inputs

        # define version of likelihood function to pass to derivative function
        def lnlike(values):
            return self.likelihood(values, self.fr_obs, self.sigma, self.rotor_model, self.obs_node, self.frequency_range)
            #return self.likelihood(values, self.x, self.data, self.sigma)

        # calculate gradients
        grads = gradients(theta, lnlike)

        outputs[0][0] = grads

#--------------------------------------------imbalance response mean function for gaussian process----------------------
# how are gradients calculated for this in HMC?
class Blackbox_Model(pm.gp.mean.Mean):
    def __init__(self, Theta):
        self.theta = Theta

    def __call__(self, x):
        # return simulator prediction given theta and X
        return simulator(self.theta, x)