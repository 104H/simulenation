
import pickle
import pandas as pd

dump_filename = 'demyelination/data/plasticity-debugging-noautapses-saveatonepsix-disableplast/other/net_'

df = pd.DataFrame()

for threadid in range(1):
    with open(dump_filename + str(threadid), "rb") as f:
        network = pickle.load(f)

        net = network['synapse_ex']

        df = pd.concat([df, net])

import pdb; pdb.set_trace()

