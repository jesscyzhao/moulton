import pandas as pd
import numpy as np


class Parameters:

    Parameters = dict()

    def add(self, param_name, value, max=None, min=None):
        # bunches of validations here!
        self.Parameters[param_name] = dict()
        self.Parameters[param_name]['value'] = value
        self.Parameters[param_name]['max'] = max
        self.Parameters[param_name]['min'] = min

        return self.Parameters

    # TODO: add multiple parameters at the same time
    # def add_list(self,  ):

