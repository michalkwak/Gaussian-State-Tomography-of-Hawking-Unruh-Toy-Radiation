import simulate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from IPython.display import Video

#scikid-tda imports
import ripser
import persim

import teaspoon.MakeData.PointCloud as makePtCloud
import teaspoon.TDA.Draw as Draw

from teaspoon.SP.network import ordinal_partition_graph
from teaspoon.TDA.PHN import PH_network
from teaspoon.SP.network_tools import make_network
from teaspoon.parameter_selection.MsPE import MsPE_tau
import teaspoon.MakeData.DynSysLib.DynSysLib as DSL

def drawPersistent(P,diagrams,R=2):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (20,5))

    # Draw point cloud
    plt.sca(axes[0])
    plt.title("Point Cloud")
    plt.scatter(P[:,0],P[:,1])

    # Draw diagrams
    plt.sca(axes[1])
    plt.title("0-dim Diagram")
    Draw.drawDgm(diagrams[0])

    plt.sca(axes[2])
    
    plt.title("1-dim Diagram")
    Draw.drawDgm(diagrams[1])
    plt.axis([0,R,0,R])

    plt.show()



# Example usage
r = 0.5  # Squeezing parameter
V = simulate.generate_covariance_matrix(r)
V_A = simulate.reduce_to_accessible(V)

n_samples = 1000
noise_level = 0.1
measurement_type = 'heterodyne'  # or 'homodyne'
rng = np.random.default_rng(seed=42)

samples = simulate.sample_measurements(V_A, n_samples, noise_level, measurement_type, rng)

D = np.cov(samples)
diagrams = ripser.ripser(D, distance_matrix=True)["dgms"]
drawPersistent(D,diagrams)