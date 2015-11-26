__author__ = 'chunyi'

import pandas as pd
import numpy as np
import copy

class HMM:

    def createRandomCluster(self, num_state, student_list):
        """
        generate a random cluster structure given number of clusters and list of unique students

        :param num_state: int, number of clusters
        :param student_list: pd.Series, series of unique students
        :return: pd.Series, indexed by student ID.
        """
        cluster = dict()
        for i in range(num_state):
            this_student = np.random.choice(student_list)
            cluster[this_student] = i
            student_list = student_list[student_list!=this_student]
        for student in student_list:
            cluster[student] = np.random.choice(range(num_state))
        return pd.Series(cluster)

    def updateRandomCluster(self, current_cluster_structure):
        """
        update cluster structure, make sure there are still num_clusters clusters in total.
        :param current_cluster_structure: pd.Series
        :return: pd.Series
        """
        counts = current_cluster_structure.value_counts()
        changeable_cluster = counts.index[counts > 1]
        this_ind = np.random.choice(current_cluster_structure.index[current_cluster_structure == np.random.choice(changeable_cluster)])
        current_cluster_structure[this_ind] = np.random.choice(counts.index)

        return current_cluster_structure

    def convertObsToStateSequence(self, obs_sequence, cluster_structure):
        """
        convert the obs_sequence to a sequence of states based on current cluster structure
        :param obs_sequence: integer or pd.timestamp indexed pd.Series
        :param cluster_structure: dict()
        :return: pd.DataFrame
        """

        assert type(obs_sequence)== pd.Series, 'observation sequence has to be a pd.Series'
        state_sequence = dict()
        for index in obs_sequence.index:
            state_sequence[index] = cluster_structure[obs_sequence[index]]
        return pd.Series(state_sequence)

    def getErrorRateMatrix(self, cluster_structure, num_state, epsilon):

        error_rate_matrix = pd.DataFrame(index=cluster_structure.index, columns=range(num_state))
        for i in range(num_state):
            error_in_this_state = pd.Series([0]*len(cluster_structure))
            error_in_this_state[cluster_structure==i] = (1-epsilon)/np.sum(cluster_structure==i)
            error_in_this_state[cluster_structure!=i] = epsilon/np.sum(cluster_structure!=i)
            error_rate_matrix[i] = error_in_this_state

        return error_rate_matrix


    def getSwitchRateMatrix(self, number_cluster, Lambda):
        pass

    def logLikelihood(self, state_sequence, num_cluster, error_rate_dict, switch_rate_dict):
        pi = [1/num_cluster]*num_cluster
        pass

    def simulateStateSequence(self, num_cluster, state_seq_lenth):
        pass

