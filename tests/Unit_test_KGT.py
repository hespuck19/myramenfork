import unittest
import os
import ramenlib as ramen
import mistlib as mist
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestLinearPhaseDiagramPt(unittest.TestCase):

    def test_linear_phase_diagram_pt(self):
        path_to_example_data = os.path.join(os.path.dirname(__file__), '../../mist/examples/AlCu.JSON')
        # Load material information using mistlib
        mat = mist.core.MaterialInformation(path_to_example_data)
        
        try:ramen.linear_phase_diagram_pt(self)
            # Call the function from ramenlib
            
        except Exception as e:
            self.fail(f"linear_phase_diagram_pt raised an exception: {str(e)}")

if __name__ == '__main__':
    unittest.main()
