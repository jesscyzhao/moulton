__author__ = 'chunyi'

import pandas as pd
import numpy as np
import copy
from scipy.stats import entropy

# TODO: screw dict, everything is either a pd.Series or pd.DataFrame
# make lambda and epsilon parameters with boundaries (0,1)
        # two layer: out layer: handel initial parameter values, making sure that initial cluster_structure are
        # different enough; handel parallelization
        # inner layer (run MLE): run a complete MLE chain with a given set of initial parameter values
"""
    Variables
        1, obs_seq: the actual/simulated single unit data (on a given day at a given dining hall)
           obs_seq_list: the list of obs_seq, all units of data (both dining hall on all days, each is iid)
        2, num_cluster: number of clusters
           cluster_structure: the cluster each individual belongs to
        3, student_list: list of unique individuals to be clustered


"""
class HMM:

    def createRandomCluster(self, num_cluster, student_list, seed=1, with_seed=False):
        """
        generate a random cluster structure given number of clusters and list of unique students
        possible values of states are 0 -- num_cluster-1
        :param num_cluster: int, number of clusters
        :param student_list: pd.Series, series of unique students
        :return: dict, with unique student ID as the key, and cluster(state) of each student as value
        """
        rng = np.random.RandomState()
        if with_seed:
            rng.seed(seed)
        cluster = dict()
        # randomly choose students to make sure that each state has at least one student in it
        for i in range(num_cluster):
            this_student = rng.choice(student_list)
            cluster[this_student] = i
            student_list = student_list[student_list != this_student]
        # the rest of the student can be randomly assigned to all states possible.
        for student in student_list:
            cluster[student] = rng.choice(range(num_cluster))
        return pd.Series(cluster)

    def updateRandomCluster(self, current_cluster_structure):
        """
        update cluster structure, make sure there are still num_clusters clusters in total.
        :param current_cluster_structure: pd.Series
        :return: pd.Series
        """
        if not isinstance(current_cluster_structure, pd.Series):
            current_cluster_structure = pd.Series(current_cluster_structure)
        counts = current_cluster_structure.value_counts()
        changeable_clusters = counts.index[counts > 1]
        cluster_to_change = np.random.choice(changeable_clusters)
        this_ind = np.random.choice(current_cluster_structure.index[current_cluster_structure == cluster_to_change])
        # Forcing it to flip to another state?
        current_cluster_structure[this_ind] = np.random.choice(counts.index)

        return current_cluster_structure

    def convertObsToStateSequence(self, obs_seq, cluster_structure):
        """
        convert the obs_seq to a sequence of states based on current cluster structure
        :param obs_seq: integer or pd.timestamp indexed pd.Series
        :param cluster_structure: dict()
        :return: pd.DataFrame
        """

        assert type(obs_seq) == pd.Series, 'observation sequence has to be a pd.Series'
        state_sequence = dict()
        for index in obs_seq.index:
            state_sequence[index] = cluster_structure[obs_seq[index]]
        return pd.Series(state_sequence)

    def getErrorRateMatrix(self, cluster_structure, num_cluster, epsilon):
        """
        error_rate_ ij is the error rate for individual i in state j
        :param cluster_structure: dict()
        :param num_cluster: int
        :param epsilon: float
        :return: dict
        """
        if not isinstance(cluster_structure, pd.Series):
            cluster_structure = pd.Series(cluster_structure)
        error_rate_matrix = pd.DataFrame(index=cluster_structure.index, columns=range(num_cluster))
        for i in range(num_cluster):
            error_in_this_state = pd.Series([0]*len(cluster_structure))
            error_in_this_state[cluster_structure==i] = (1-epsilon)/np.sum(cluster_structure==i)
            error_in_this_state[cluster_structure!=i] = epsilon/np.sum(cluster_structure!=i)
            error_rate_matrix[i] = error_in_this_state
        return error_rate_matrix


    def getSwitchRateMatrix(self, num_cluster, Lambda):
        """
        Switch rate is define as 1 - Lambda if i and j are the same state, Lambda/(K-1) otherwise
        :param number_cluster: int, K
        :param Lambda: float
        :return:
        """
        switch_rate_matrix = pd.DataFrame(index=range(num_cluster), columns=range(num_cluster))
        for i in range(num_cluster):
            switch_rate_matrix.loc[i] = [Lambda/(num_cluster-1)] * num_cluster
            switch_rate_matrix.loc[i][i] = 1-Lambda
        return switch_rate_matrix


    def forward(self, obs_seq, num_cluster, error_rate_dict, switch_rate_dict):

        pi = [1/num_cluster] * num_cluster
        path_length = len(obs_seq)

        alpha_dict = dict()

        alpha_dict[0] = dict(zip(range(num_cluster), [pi[i] * error_rate_dict[i][obs_seq[0]]
                                                              for i in range(num_cluster)]))

        for t in range(path_length - 1):
            alpha_dict[t+1] = dict()
            for j in range(num_cluster):
                test = list()
                for i in range(num_cluster):
                    test_1 = alpha_dict[t][i]
                    switch = switch_rate_dict[i][j]
                    test.append(test_1 * switch)
                # print([alpha_dict[t][i] * switch_rate_dict[i][j] for i in range(num_cluster)])
                sum_alpha_j = np.sum(test)
                alpha_dict[t+1][j] = sum_alpha_j * error_rate_dict[j][obs_seq[t+1]]
        alpha_df = pd.DataFrame.from_dict(alpha_dict)

        final = np.sum(alpha_df.ix[:, path_length-1])

        return np.log(final)

    def logLikelihood(self, obs_seq, num_cluster, error_rate_dict, switch_rate_dict):
        """

        :param obs_seq: state list of the same length as observation sequence
        :param num_cluster: int
        :param error_rate_dict: dict
        :param switch_rate_dict: dict
        :return: log likelihood
        """
        pi = [1/num_cluster] * num_cluster
        path_length = len(obs_seq)

        alpha_dict = dict()

        alpha_dict[0] = dict(zip(range(num_cluster), [np.log(pi[i]) + np.log(error_rate_dict[i][obs_seq[0]])
                                                      for i in range(num_cluster)]))

        for t in range(path_length-1):
            alpha_dict[t+1] = dict()
            for j in range(num_cluster):
                log_alpha_j = [(alpha_dict[t][i] + np.log(switch_rate_dict[i][j])) for i in range(num_cluster)]
                max_log_alpha_j = max(log_alpha_j)
                sum_residual = np.sum(np.exp(log_alpha_j - max_log_alpha_j))
                final = max_log_alpha_j + np.log(sum_residual)
                alpha_dict[t+1][j] = final + np.log(error_rate_dict[j][obs_seq[t + 1]])

        alpha_df = pd.DataFrame.from_dict(alpha_dict)
        max_final = max(alpha_df.ix[:, path_length-1])

        llk = max_final + np.log(np.sum(np.exp(alpha_df.ix[:, path_length-1] - max_final)))

        return llk

    def proposeLambdaOrEpsilon(self, current_value):
        new_value = np.random.uniform(0, 1)
        if new_value == current_value or new_value == 0:
            new_value = np.random.uniform(0, 1)
        return new_value

    def singleChainMLE(self,
                       obs_seq_list,
                       num_cluster,
                       init_params,
                       num_iter=10000,
                       thin=100):
        Lambda = init_params['Lambda']['value']
        epsilon = init_params['epsilon']['value']
        current_cluster_structure = init_params['cluster']['value']

        mle_chain = dict()

        error_rate_dict = self.getErrorRateMatrix(current_cluster_structure, num_cluster, epsilon)
        switch_rate_dict = self.getSwitchRateMatrix(num_cluster, Lambda)
        current_llk = np.sum([self.logLikelihood(x, num_cluster, error_rate_dict, switch_rate_dict) for x in
                                            obs_seq_list])

        mle_chain[0] = { 'Lambda': copy.deepcopy(Lambda),
                         'epsilon': copy.deepcopy(epsilon),
                         'cluster_structure': copy.deepcopy(current_cluster_structure),
                         'current_llk': copy.deepcopy(current_llk)
                        }
        print(mle_chain[0])
        for i in range(1, num_iter):
            coin = np.random.uniform(0, 1)
            new_Lambda = copy.deepcopy(Lambda)
            new_epsilon = copy.deepcopy(epsilon)
            new_cluster_structure = copy.deepcopy(current_cluster_structure)
            if coin < 1/3:
                new_Lambda = self.proposeLambdaOrEpsilon(Lambda)
            elif coin < 2/3:
                new_epsilon = self.proposeLambdaOrEpsilon(epsilon)
            # else:
            #     new_cluster_structure = self.updateRandomCluster(current_cluster_structure)

            error_rate_dict = self.getErrorRateMatrix(new_cluster_structure, num_cluster, new_epsilon)
            switch_rate_dict = self.getSwitchRateMatrix(num_cluster, new_Lambda)

            propose_llk = np.sum([self.logLikelihood(x, num_cluster, error_rate_dict, switch_rate_dict) for x in
                                 obs_seq_list])

            if propose_llk > current_llk:
                Lambda = copy.deepcopy(new_Lambda)
                epsilon = copy.deepcopy(new_epsilon)
                current_cluster_structure = copy.deepcopy(new_cluster_structure)
                current_llk = copy.deepcopy(propose_llk)

            mle_chain[i] = {'Lambda': copy.deepcopy(Lambda),
                            'epsilon': copy.deepcopy(epsilon),
                            'cluster_structure': copy.deepcopy(current_cluster_structure),
                            'current_llk': copy.deepcopy(current_llk)
                            }

            if i % thin == 0:
                print(mle_chain[i])

        return mle_chain

    def multiChianMLE(self,
                      obs_seq_list,
                      num_cluster,
                      init_params,
                      num_iter= 10000,
                      thi=100
                        ):

        pass

    def generateData(self, num_cluster, obs_seq_lenth, cluster_structure, epsilon, Lambda, seed=1):
        """
        Use seed so to make sure that each time the data set is the same.
        """
        assert num_cluster == len(set(cluster_structure)), "Missing cluster in the cluster_structure"

        rng = np.random.RandomState()
        rng.seed(1)

        individuals = cluster_structure.index
        error_rate_dict = self.getErrorRateMatrix(cluster_structure, num_cluster, epsilon)
        switch_rate_dict = self.getSwitchRateMatrix(num_cluster, Lambda)

        pi = [1/num_cluster] * num_cluster

        sim_states = [0] * obs_seq_lenth
        sim_data = [0] * obs_seq_lenth

        sim_states[0] = rng.choice(range(num_cluster), p=pi)
        sim_data[0] = rng.choice(individuals, replace=False, p=error_rate_dict[sim_states[0]])

        for i in range(1, obs_seq_lenth):
            last_state = copy.deepcopy(sim_states[i-1])
            clusters = list(range(num_cluster))
            switch_rates = list(copy.deepcopy(switch_rate_dict[last_state]))
            sim_states[i] = rng.choice(clusters, replace=False, p=switch_rates)
            sim_data[i] = rng.choice(individuals, replace=False, p=list(error_rate_dict[sim_states[i]]))

        return {'sim_data': sim_data, 'sim_states': sim_states}

    def KL_divergence(self, estimated_cluster_structure, actual_cluster_structure, num_cluster):
        return entropy(estimated_cluster_structure, actual_cluster_structure)





