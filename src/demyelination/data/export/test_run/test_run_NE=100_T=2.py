import sys
sys.path.append('/home/pbr-student/simulenation/src/demyelination')
sys.path.append('/home/pbr-student/simulenation/src')
import matplotlib
matplotlib.use('Agg')
from fna.tools.parameters import *
from fna.tools.analysis import *
from experiments import ai_run

ai_run.run('./test_run_NE=100_T=2.txt', **{})