import sys
sys.path.append('/users/hameed/simulenation/src/demyelination')
sys.path.append('/users/hameed/simulenation/src')
import matplotlib
matplotlib.use('Agg')
from fna.tools.parameters import *
from fna.tools.analysis import *
from experiments import ai_run

ai_run.run('./exp1_nuX=12_gamma=9_nTRN=500.txt', **{})