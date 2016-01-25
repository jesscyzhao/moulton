from HMM.Model import HMM

import pandas as pd
from HMM.Util import Parameters

def main():
    model = HMM()
    K = 3
    Lambda = 0.1
    student_list = pd.Series(range(5))
    random_cluster = model.createRandomCluster(K, student_list, with_seed=True)

    print('init_cluster', random_cluster)
    obs_sequence = pd.Series([0, 1, 2, 3, 4])

    # state_sequence = model.convertObsToStateSequence(obs_sequence, update_cluster)

    error_dict = model.getErrorRateMatrix(random_cluster, K, 0.1)

    switch_dict = model.getSwitchRateMatrix(K, Lambda)

    print(error_dict)
    print(switch_dict)
    forward_llk = model.forward(obs_sequence, K, error_dict, switch_dict)
    test_llk = model.logLikelihood(obs_sequence, K, error_dict, switch_dict)

    print('forward llk', forward_llk)
    print('llk', test_llk)

    # init_params = dict()
    # init_params['Lambda'] = {'value': 0.1}
    # init_params['epsilon'] = {'value': 0.1}
    # init_params['cluster'] = {'value': random_cluster}

    # num_iter = 10
    # thin = 1
    # chain = model.singleChainMLE([obs_sequence], K, init_params, num_iter, thin)
    #
    # # Test the data generator
    # model = HMM()
    # Lambda = 0.5
    # epsilon = 0.05
    # K=2
    # num_unique_students = 20
    # true_cluster_structure = model.createRandomCluster(K, pd.Series(range(num_unique_students)))
    # obs_seq_lenth = 100
    # sim_data = model.generateData(K, obs_seq_lenth, true_cluster_structure, epsilon, Lambda, seed=1)
    # sim_data_2 = model.generateData(K, obs_seq_lenth, true_cluster_structure, epsilon, Lambda, seed=1)
    #
    # print(sim_data)
    # print(sim_data_2)

if __name__ == '__main__':
    main()