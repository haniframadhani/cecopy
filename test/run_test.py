import unittest
import sys
import os

# Add the parent directory of 'ceco' to the Python path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))

# Discover and run all tests in the current directory
if __name__ == "__main__":
    loader = unittest.TestLoader()

    # Adjust the start_dir to point to the correct subdirectories
    suite = loader.discover(start_dir='test', pattern='test*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
