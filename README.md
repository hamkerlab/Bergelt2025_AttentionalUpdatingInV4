# Bergelt2025_AttentionalUpdatingInV4

The repository contains all relevant scripts to reproduce the results and figures published in the article
"Perisaccadic attentional updating in Area V4: A neuro-computational approach" by J. Bergelt and F. H. Hamker, submitted in European Journal in Neuroscience

The source code is implemented in Python 3.10.  
The simulations were done with the neural simulator ANNarchy 4.8.

At first, a set of trials must be created using the script generatingSetups.py. Afterwards, these trials can be simulated in parallel with ANNarchy using the script main.py. With main_singleTrial.py, a single trial with more recordings can be simulated (the index of the trial can be specified).  
The neuro-computational model is defined in the scripts under subfolder "network".  
The different parameters defining the experimental setup and the neuro-computational model as well as some general parameters are stored in subfolder "parameters".  
To create the figures, run evaluate.py (in subfolder "evaluation").  
With script run.sh all the steps are run in the correct order at once.

## Structure of the repository
  * evaluation
    * evaluate.py (main script for generating results)
    * plotting.py (generates figures)
    * processData.py (pre-processing simulation results, needed for plotting)
  * network
    * connections.py (defines own connection pattern)
    * helper.py (auxiliary functions)
    * model.py (defines ANNarchy model including neurons, populations and connections)
  * parameters
    * params_general.py (defining general parameters like where to save, number of trials, ...)
    * params_model.py (defining model parameters including neuronal, connection and input parameters, layer sizes, ...)
    * param_plotting.py (defining parameters used for generating figures)
  * generateSetup.py (generates setup for all trials including stimulus inputs and eye movement, run in advance)
  * generateSignals.py (generates input signals out of given setup)
  * helper.py (auxiliary functions)
  * main_singleTrial.py (simulate one (specified) trial and record many data)
  * main.py (simulate all trials and record only needed data)
  * run.sh (script running all files in correct order at once)
  * SaccGen.py (generates saccade after Van Wetter & Van Opstal (2008))
  * saving.py (auxiliary functions for saving data)

## Reproducibility
The neurocomputational model itself is deterministic. Only the different trials are generated randomized. Thus, running the scripts will produce nearly the same results as presented in the paper. 

Nevertheless, if you want to reproduce identical results compared to the ones of the paper, please use the provided data in the releases.


## Dependencies

Neural Simulator ANNarchy 4.8.2  
python 3.10, numpy 1.26.4, scipy 1.14.1, matplotlib 3.9.2, h5py 3.12.1, Cython 3.0.11
