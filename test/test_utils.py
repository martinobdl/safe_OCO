import unittest
import numpy as np
import utils


class TestProject(unittest.TestCase):
    def test_project_array(self):
        array = utils.project(np.random.randn(100))
        self.assertAlmostEqual(array.sum(), 1, 6, "Not one!")
        self.assertGreater(np.min(array), -1e-7)
        self.assertLess(np.max(array), 1+1e-7)


if __name__ == '__main__':
    unittest.main()
