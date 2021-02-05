from unittest import TestCase
import os

from syndalib.loader import load_adelaide


class Test(TestCase):
    def test_load_adelaide(self):
        print(os.getcwd())
        ldata, llabels = load_adelaide()
        data = ldata[0]
        labels = llabels[0]
        print(data.shape)
        assert True
