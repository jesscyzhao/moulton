
import pandas as pd
import numpy as np

class HMMEntity:

    def __init__(self, name, raw_data, num_cluster):

        self.name = name
        self.raw_data = raw_data
        self.num_cluster = num_cluster

    def _setStudentList(self):
        pass

    def _setObsSequenceData(self):
        pass

    def _setStateSequenceData(self, cluster_structure):
        pass

    def getStudentList(self):
        pass

    def getObsSequenceData(self, day, hall):
        pass
