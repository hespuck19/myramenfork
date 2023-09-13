import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

import mist
import ramen

# Create from the AlCu JSON file
path_to_example_data = os.path.join(os.path.dirname(__file__), 'AlCu.json')
mat = mist.core.MaterialInformation(path_to_example_data)

velocity = 0.1 # m/s
c_avg = 2.9 # at. %
phases = ['alpha', 'theta']
spacing = ramen.get_eutectic_lamellar_spacing(mat, velocity, c_avg, phases)
print("Lamellar spacing: ", spacing*1.e9, "nm")