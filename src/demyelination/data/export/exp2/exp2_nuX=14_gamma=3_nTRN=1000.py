import sys
sys.path.append('/users/hameed/simulenation/src/demyelination')
sys.path.append('/users/hameed/simulenation/src')
import matplotlib
matplotlib.use('Agg')
from fna.tools.parameters import *
from fna.tools.analysis import *
from experiments import ai_run

ai_run.run('./exp2_nuX=14_gamma=3_nTRN=1000.txt', **{})