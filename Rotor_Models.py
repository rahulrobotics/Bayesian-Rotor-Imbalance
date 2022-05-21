import ross as rs
import numpy as np
import math
from scipy.optimize import least_squares 
import pymc3 as pm
import arviz as az
import theano.tensor as tt
from rotor_models_BI_helper import *
import pickle
import matplotlib.pyplot as plt


#--------------------------------------- define "real" rotor and simulated rotor----------------------------------------

steel = rs.Material(name="Steel", rho=7810, E=211e9, G_s=81.2e9)

n = 2

shaft_elem = [
    rs.ShaftElement(
        L=0.25,
        idl=0.0,
        odl=0.05,
        material=steel,
        shear_effects=True,
        rotary_inertia=True,
        gyroscopic = True
    )
    for _ in range(n)
]

disk0 = rs.DiskElement.from_geometry(
    n=1, material=steel, width=0.07, i_d=0.05, o_d=0.28
)

disks = [disk0]

stfx_real = 1e6
stfy_real = 0.8e6
dampx_real = 3e3
dampy_real = 5e3
bearing0_real = rs.BearingElement(0, kxx=stfx_real, kyy=stfy_real, cxx = dampx_real, cyy = dampy_real)
bearing1_real = rs.BearingElement(2, kxx=stfx_real, kyy=stfy_real, cxx = dampx_real, cyy = dampy_real)

bearings_real = [bearing0_real, bearing1_real]

rotor_real = rs.Rotor(shaft_elem, disks, bearings_real)

stfx_sim = 1e6
stfy_sim = 1e6
bearing0_sim = rs.BearingElement(0, kxx=stfx_sim, kyy=stfy_sim, cxx = 0, cyy = 0)
bearing1_sim = rs.BearingElement(2, kxx=stfx_sim, kyy=stfy_sim, cxx = 0, cyy = 0)

bearings_sim = [bearing0_sim, bearing1_sim]

rotor_sim = rs.Rotor(shaft_elem, disks, bearings_sim)

# plotting rotor model

fig_rotor_real = rotor_real.plot_rotor()
# fig_rotor_real.write_image("rotor_real.png")
#---------------------------------------- important variables-----------------------------------------------------------
SAMPLES = 61
FREQUENCY_RANGE = np.linspace(315, 1150, SAMPLES)
DISK_NODE = 1
SIGMA_TRUE = 0.01
IMBALANCE_MAGNITUDE = 0.03
IMBALANCE_PHASE = 0

#---------------------------------------- real imbalance response-------------------------------------------------------
samples = 61
results_real, fr_real = getImbalanceResponse(rotor_real, DISK_NODE, IMBALANCE_MAGNITUDE, IMBALANCE_PHASE, FREQUENCY_RANGE)

# noise
epsilon = np.random.randn(4, SAMPLES) * SIGMA_TRUE
observations = fr_real + epsilon

# plot real response
probe_real = (1, 45)
fig_results_real = results_real.plot(probe = [probe_real], probe_units="degrees")
fig_results_real.write_image("fr_real.png")

#-------------------------------------- simulated imbalance response----------------------------------------------------
results_sim, fr_sim = getImbalanceResponse(rotor_sim, DISK_NODE, IMBALANCE_MAGNITUDE, IMBALANCE_PHASE, FREQUENCY_RANGE)

# plot sim response
probe_sim = (1, 45)
fig_results_sim = results_sim.plot(probe = [probe_sim], probe_units="degrees")
fig_results_sim.write_image("fr_sim.png")
#-------------------------------------- least-squares formulation-------------------------------------------------------








# #------------------------- Bayesian Imbalance Identification (no discrepancy)-------------------------------------------
#
# # create op
# imb_resp_op = RotorModelOp(get_imb_resp=getImbalanceResponse, rotor_model=rotor_sim, obs_node=DISK_NODE, frequency_range=FREQUENCY_RANGE)
# rotor_model = pm.Model()
#
# with rotor_model:
#     # Priors for unknown model parameters
#     mag_rand = pm.HalfNormal('mag_rand', sigma=0.4) # magnitude is always positive
#     phase_rand = pm.Uniform('phase_rand', lower = 0, upper = 2*math.pi ) # radians
#
#     Beta = tt.as_tensor_variable([mag_rand, phase_rand])
#
#     # Expected value of outcome
#     # pm.Potential("likelihood", logl(Beta))
#
#     mu = imb_resp_op(Beta)
#
#     # Likelihood (sampling distribution) of observations
#     Y_obs = pm.Normal("Y_obs", mu=mu, sigma=SIGMA_TRUE, observed=observations)
#
#     # trace = pm.sample(1000, tune = 1000, chains = 2, return_inferencedata=False, cores=1)
#
#     # load trace from pickle
#     print('loading trace')
#
#     trace = pickle.load(open("trace_no_discrepancy.p", "rb"))
#
#     idata = az.from_pymc3(trace_thin1, log_likelihood=False)
#
#
#     print('summarizing trace')
#     print(az.summary(idata, round_to=2))
#     print('plotting trace')
#     az.plot_trace(idata)
#     plt.show()


#------------------------- Bayesian Imbalance Identification (with discrepancy)-----------------------------------------

# create op
imb_resp_op = RotorModelOp(get_imb_resp=getImbalanceResponse, rotor_model=rotor_sim, obs_node=DISK_NODE, frequency_range=FREQUENCY_RANGE)
rotor_model = pm.Model()

X = FREQUENCY_RANGE.reshape((-1,1))

with rotor_model:
    # gaussian process hyper-parameters
    length_scale = pm.Bound(pm.Gamma, upper = 4)("length_scale", alpha=5, beta=5) # taken from discrepancy modeling paper
    eta = pm.HalfCauchy('eta', beta=5)

    # define covariance matrix
    cov_func = (eta**2) * pm.gp.cov.ExpQuad(1, ls=length_scale)
    Sigma = cov_func(X) + np.eye(SAMPLES)*SIGMA_TRUE**2 # gaussian process covariance + noise covariance

    # Priors for unknown model parameters
    mag_rand = pm.HalfNormal('mag_rand', sigma=0.4) # magnitude is always positive
    phase_rand = pm.Uniform('phase_rand', lower = 0, upper = 2*math.pi ) # radians

    Beta = tt.as_tensor_variable([mag_rand, phase_rand])
    mu = imb_resp_op(Beta)

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.MvNormal("Y_obs", mu=mu, cov=Sigma, observed=observations)

    step = pm.Metropolis()
    trace = pm.sample(1, tune = 1, step = step, chains = 2, return_inferencedata=False, cores=1)

    ## load trace from pickle
    # print('loading trace')
    # trace = pickle.load(open("trace_no_discrepancy.p", "rb"))
    # idata = az.from_pymc3(trace_thin1, log_likelihood=False)
    # print('summarizing trace')
    # print(az.summary(idata, round_to=2))
    # print('plotting trace')
    # az.plot_trace(idata)
    # plt.show()

