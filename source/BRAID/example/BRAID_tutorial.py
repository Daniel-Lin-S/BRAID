import logging
import os
import sys

sys.path.insert(0, os.path.join("..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Setting up logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import coloredlogs

coloredlogs.install(
    level=logging.INFO,
    fmt="%(asctime)s %(name)s [%(filename)s > %(lineno)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    logger=logger,
)


# Prepare plots
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

matplotlib.rcParams["figure.facecolor"] = "w"
matplotlib.rcParams["axes.facecolor"] = "w"
matplotlib.rcParams.update(
    {"font.size": 16, "axes.titlesize": 14, "axes.labelsize": 14}
)

# Import BRAID tools
import BRAID
from BRAID.BRAIDModel import BRAIDModel, runPredict
from BRAID.MainModel import MainModel
from BRAID.config import get_default_config_path, load_braid_fit_arguments
from BRAID.tools.evaluation import computeEigIdError, evalPrediction, evaluateDecoding
from BRAID.tools.file_tools import pickle_load
from BRAID.tools.LSSM import LSSM
from BRAID.tools.SSM import SSM

# Set random seeds for exact reproducibility
seed = 42
import random

import numpy as np
import tensorflow as tf

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)



##################################################################################
######## Example 1: Full BRAID example including preprocessing
#                   on simulated data with nonlinear behaviour decoder
##################################################################################
# Load data
# The simulated data and its true generative model are loaded from the data directory.
data_dir = os.path.join(os.path.dirname(BRAID.__file__), "..", "..", "data")
sample_data_file_name_1 = "sample_data_CzNL_prep.p"
sample_data_path1 = os.path.join(data_dir, sample_data_file_name_1)

print("Loading example data from {}".format(sample_data_path1))
data1 = pickle_load(sample_data_path1)

maxN = int(2 * 1e4)  # Number of data points to use in this notebook
N = np.min((maxN, data1["x"].shape[0]))
x = data1["x"][:N, :] # get data
y = data1["y"][:N, :]
z = data1["z"][:N, :]
u = data1["u"][:N, :]
trueLSSM = LSSM(params=data1['csys']) 
trueModel_main = SSM(lssm=trueLSSM, params=data1['csys']) # Create the object containing the main true model corresponding to [X1, X2]^T
trueModel_beh = LSSM(params=data1['csys']['zErrSys']) # Create the object that contains the true model corresponding to X3 (behaviour-specific dynamics)
nx = trueModel_main.state_dim  # Dimesnionality of the main model (true n1+n2)
n1 = len(trueModel_main.zDims)  # The number of encoded latent states in the simulated model that drive behaviour (behaviourally relevant neural dynamics)

allYData, allZData, allUData = y, z, u

# Separate data into training and test data:
testInds = np.arange(np.round(0.5 * allYData.shape[0]), dtype=int)
trainInds = np.arange(1 + testInds[-1], allYData.shape[0])
yTrain = allYData[trainInds, :]
yTest = allYData[testInds, :]
zTrain = allZData[trainInds, :]
zTest = allZData[testInds, :]
uTrain = allUData[trainInds, :]
uTest = allUData[testInds, :]

# Normalize data (removing the mean is essential before training):
yMean, yStd = np.mean(yTrain,axis=0), np.std(yTrain,axis=0)
yTrainN, yTestN = (yTrain - yMean)/yStd, (yTest - yMean)/yStd

zMean, zStd = np.mean(zTrain,axis=0), np.std(zTrain,axis=0)
zTrainN, zTestN = (zTrain - zMean)/zStd, (zTest - zMean)/zStd

uMean, uStd = np.mean(uTrain,axis=0), np.std(uTrain,axis=0)
uTrainN, uTestN = (uTrain - uMean)/uStd, (uTest - uMean)/uStd

# Fit BRAID model to data:

idSysF = BRAIDModel()

config_path = get_default_config_path('behaviour_decoder_mlp_1x64.yaml')
fit_arguments = load_braid_fit_arguments(config_path)

# The YAML dimensions define the neural and non-neural state sizes.
idSysF.fit(yTrainN.T, Z=zTrainN.T, U=uTrainN.T, **fit_arguments)
idSysF.yMean, idSysF.yStd,  idSysF.zMean, idSysF.zStd,  idSysF.uMean, idSysF.uStd =  yMean, yStd,  zMean, zStd,  uMean, uStd # saving preprocessing info in model objects to use during inference in runPredict function
print('Training finished after {} epochs'.format(len(idSysF.sId.logs['model1']['history']['loss'])))

# Inference:
# Predict behaviour and neural activity, from past neural activity and inputs, using the learned model
print("Computing predictions for the learned model")
# This function takes test data (unnormalized). Normalization happens inside runPredict given the saved mean and std attributes
zTestPredF, yTestPredF, xTestPredF, _, _, _ = runPredict(idSysF, yTest, zTest, uTest, YType='cont', ZType='cont')
# runPredict function outputs are lists of length= #steps_ahead desired for inference which is taken from the field: idSysF.step_ahead
# Here we are only looking at 1-step-ahead predictions i.e., the lists only have one element:
zTestPredF, yTestPredF, xTestPredF = zTestPredF[0], yTestPredF[0], xTestPredF[0] 

# Compute CC of decoding and self-prediction
zCCNonLinF = evalPrediction(zTestN, zTestPredF, "CC")
yCCNonLinF = evalPrediction(yTestN, yTestPredF, "CC")

# Decode behaviour and do neural self-prediction using the true model for comparison
print("Computing predictions for the true model (slower due to the analytical symbolic implementation of nonlinearity in the true model)")
zTestPredIdeal, yTestPredIdeal, xTestPredIdeal, _, _, _ = runPredict(trueModel_main, yTest, zTest, uTest, YType='cont', ZType='cont')
zTestPredIdeal, yTestPredIdeal, xTestPredIdeal = zTestPredIdeal[0], yTestPredIdeal[0], xTestPredIdeal[0] 

zCCIdeal = evalPrediction(zTest, zTestPredIdeal, "CC")
yCCIdeal = evalPrediction(yTest, yTestPredIdeal, "CC")

print(
    f"Behaviour decoding CC:\n  BRAID => {np.mean(zCCNonLinF):.3g}, Ideal using true model => {np.mean(zCCIdeal):.3g}"
)
print(
    f"Neural self-prediction CC:\n  BRAID => {np.mean(yCCNonLinF):.3g}, Ideal using true model => {np.mean(yCCIdeal):.3g}"
)

print('End of example 1!')


######################################################################################
################## Example 2: BRAID forecasting example on simulated data 
#                             with nonlinear behaviour decoder
######################################################################################

# Load data
# The simulated data and its true generative model are loaded from the data directory.
data_dir = os.path.join(os.path.dirname(BRAID.__file__), "..", "..", "data")
sample_data_file_name_2 = "sample_data_CzNL.p"
sample_data_path2 = os.path.join(data_dir, sample_data_file_name_2)

print("Loading example data from {}".format(sample_data_path2))
data2 = pickle_load(sample_data_path2)
# This data is from an example model from results shown in Fig. 2a-d, with nonlinearity only in the behaviour decoder.

maxN = int(2 * 1e4)  # Number of data points to use in this notebook
N = np.min((maxN, data2["x"].shape[0]))
x = data2["x"][:N, :]
y = data2["y"][:N, :]
z = data2["z"][:N, :]
u = data2["u"][:N, :]

trueLSSM = LSSM(params=data2['csys'])
trueModel_main = SSM(lssm=trueLSSM, params=data2['csys']) #data["csys"]  # The object that contains the true corresponding to [X1, X2]^T

trueLSSM = LSSM(params=data2['csys'])
trueModel = SSM(lssm=trueLSSM, params=data2['csys']) 
nx = trueModel.state_dim  # Total number of latent states in the simulated model
n1 = len(trueModel.zDims)  # The number of encoded latent states in the simulated model that drive behaviour (behaviourally relevant neural dynamics)

allYData, allZData, allUData = y, z, u

# Separate data into training and test data:
testInds = np.arange(np.round(0.5 * allYData.shape[0]), dtype=int)
trainInds = np.arange(1 + testInds[-1], allYData.shape[0])
yTrain = allYData[trainInds, :]
yTest = allYData[testInds, :]
zTrain = allZData[trainInds, :]
zTest = allZData[testInds, :]
uTrain = allUData[trainInds, :]
uTest = allUData[testInds, :]

# Normalize data:
yMean, yStd = np.mean(yTrain,axis=0), np.std(yTrain,axis=0)
yTrainN, yTestN = (yTrain - yMean)/yStd, (yTest - yMean)/yStd

zMean, zStd = np.mean(zTrain,axis=0), np.std(zTrain,axis=0)
zTrainN, zTestN = (zTrain - zMean)/zStd, (zTest - zMean)/zStd

uMean, uStd = np.mean(uTrain,axis=0), np.std(uTrain,axis=0)
uTrainN, uTestN = (uTrain - uMean)/uStd, (uTest - uMean)/uStd

# Fit BRAID model to data:
idSys = BRAIDModel()
config_path = get_default_config_path(
    'behaviour_decoder_mlp_1x64_observed_input_forecast.yaml'
)
fit_arguments = load_braid_fit_arguments(config_path)
idSys.fit(yTrainN.T, Z=zTrainN.T, U=uTrainN.T, **fit_arguments)
idSysF = idSys.sId
idSysF.yMean, idSysF.yStd,  idSysF.zMean, idSysF.zStd,  idSysF.uMean, idSysF.uStd =  yMean, yStd,  zMean, zStd,  uMean, uStd
print('Training finished after {} epochs'.format(len(idSysF.logs['model1']['history']['loss'])))

# Inference:
steps_ahead = [1,2,4,8,16,32] # steps_ahead we want to evaluate the learned model on
idSysF.set_steps_ahead(steps_ahead)
idSysF.zDims = np.arange(1, 1+min([n1, nx]))
# Predict behaviour and neural activity, from past neural activity and inputs, using the learned model
print("Computing predictions for the learned model")
zTestPredF, yTestPredF, xTestPredF, _, _, _ = runPredict(idSysF, yTest, zTest, uTest, YType='cont', ZType='cont')
# runPredict function outputs are lists of length= #steps_ahead desired for inference which is taken from the field: idSysF.step_ahead

# Compute CC of decoding and self-prediction
zCCNonLinF, yCCNonLinF = np.nan*np.ones((len(steps_ahead),1)), np.nan*np.ones((len(steps_ahead),1))
for sai, step in enumerate(steps_ahead):
    zCCNonLinF[sai] = evalPrediction(zTestN[step-1:,:], zTestPredF[sai][step-1:,:], "CC")
    yCCNonLinF[sai] = evalPrediction(yTestN[step-1:,:], yTestPredF[sai][step-1:,:], "CC")

# Decode behaviour and do neural self-prediction using the true model for comparison
print(
    "Computing predictions for the true model (slower due to the analytical symbolic implementation of nonlinearity in the true model)"
)
trueModel.steps_ahead = steps_ahead
zTestPredIdeal, yTestPredIdeal, xTestPredIdeal, _, _, _ = runPredict(trueModel, yTest, zTest, uTest, YType='cont', ZType='cont') # True model accepts non-normalized data
zCCIdeal, yCCIdeal = np.nan*np.ones((len(steps_ahead),1)), np.nan*np.ones((len(steps_ahead),1))
for sai, step in enumerate(steps_ahead):
    zCCIdeal[sai] = evalPrediction(zTest[step-1:,:], zTestPredIdeal[sai][step-1:,:], "CC")
    yCCIdeal[sai] = evalPrediction(yTest[step-1:,:], yTestPredIdeal[sai][step-1:,:], "CC")

# Plot forecasting
plt.figure(figsize=(6,7))
ax = plt.axis()
xlabel = 'log2 # step ahead'
plt.subplot(2,1,1)
plt.plot(np.log2(np.array(steps_ahead)), zCCNonLinF, label='BRAID behaviour decoder')
plt.plot(np.log2(np.array(steps_ahead)), zCCIdeal, label='Ideal')
plt.legend(bbox_to_anchor=(0.5, 1.8), loc='upper center')
plt.xlabel(xlabel)
plt.ylabel('Behaviour decoding CC')
plt.grid(linewidth=0.5, alpha=0.5)
plt.ylim([0,1])

plt.subplot(2,1,2)
plt.plot(np.log2(np.array(steps_ahead)), yCCNonLinF, label='BRAID behaviour decoder')
plt.plot(np.log2(np.array(steps_ahead)), yCCIdeal, label='Ideal')
plt.legend(bbox_to_anchor=(0.5, 1.8), loc='upper center')
plt.xlabel(xlabel)
plt.ylabel('Neural self-prediction CC')
plt.grid(linewidth=0.5, alpha=0.5)
plt.subplots_adjust(left=0.2,bottom=0.1, right=0.8, top=0.7, wspace=0.1, hspace=0.3)
plt.ylim([0,1])

# Get intrinsic behavioural relevant dynamics eigenvalues and calculated normalized eigenvalue error:
Afw_true = trueModel.A
Afw = idSysF.model1.rnn.cell.Afw.get_weights()[0].T
idZEigs = np.linalg.eig(Afw[list(idSysF.zDims-1)][:, list(idSysF.zDims-1)])[0]
trueZEigs = np.linalg.eig(Afw_true)[0]
zEigErr_Afw = computeEigIdError(trueZEigs, [idZEigs], 'NFN')[0]


print('End ... !')