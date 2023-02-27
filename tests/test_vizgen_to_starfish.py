import unittest
from pathlib import Path, PosixPath

import numpy as np
import pandas as pd

from . import vizgen_to_starfish
from starfish import Codebook


class TestVizgenToStarfish(unittest.TestCase):
    """Test class for vizgen_to_starfish tool"""

    @classmethod
    def setUpClass(cls):
        cls.RAW_DIR = (
            "/mnt/ibm_lg/merscope/merfish_raw_data/"
            "202205191322_MsKidney-wtK3-VZG116b-VT-061922_VMSC02401"
        )

        cls.ANALYSIS_DIR = (
            "/mnt/ibm_lg/merscope/merfish_output/"
            "202205191322_MsKidney-wtK3-VZG116b-VT-061922_VMSC02401"
        )

    def setUp(self):
        """ """
        self.experiment = vizgen_to_starfish.VizgenExperiment(
            raw_dir=self.RAW_DIR, analysis_dir=self.ANALYSIS_DIR
        )

    def test_load_csv_codebook(self):
        """ """
        assert self.experiment.codebook is not None
        assert isinstance(self.experiment.codebook, Codebook)

    def test_load_dataorganization(self):
        """ """
        assert isinstance(self.experiment.vizgen_data, pd.DataFrame)
        assert len(self.experiment.vizgen_data) > 0

    def test_load_experiment(self):
        """ """
        assert isinstance(self.experiment.data, pd.DataFrame)
        assert len(self.experiment.data) > 0


# def test_add(self): # write a different test function for each module function
# 	# write an assert for each edge case for each function
# 	self.assertEqual(my_calc.add(10, 5), 15)
# 	self.assertEqual(my_calc.add(-1, 1), 0)
# 	self.assertEqual(my_calc.add(-1, -1), -2)

if __name__ == "__main__":
    unittest.main()
