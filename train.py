import model
import numpy as np
import h5py
import sys

d_file = h5py.File(sys.argv[1], 'r')

weights, sess = model.train(d_file, model.weights, 64)

w_file = h5py.File(sys.argv[2], 'w')
for name in model.weight_names:
    w_file.create_dataset(name, data=np.array(weights[name].eval(session=sess)))
