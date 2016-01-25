from HMM.Model import HMM
import pandas as pd

if __name__ == '__main__':

    # generate sim data
    model = HMM()
    Lambda = 0.2
    epsilon = 0.01
    K = 3
    num_unique_students = 20
    true_cluster_structure = model.createRandomCluster(K, pd.Series(range(num_unique_students)), with_seed=True)
    obs_seq_len = 100
    sim_data = model.generateData(K, obs_seq_len, true_cluster_structure, epsilon, Lambda, seed=1)

    test_data = pd.DataFrame({'obs_seq':sim_data['sim_data']})
    output_true_cluster_structure = pd.DataFrame({'clust_struct':true_cluster_structure})
    test_data.to_csv('python_sim_data_seed_1.csv', sep=',', index=False)
    output_true_cluster_structure.to_csv('python_cluster_structure_seed_1.csv', sep=',', index=False)

    # initialization
    init_cluster_structure = model.createRandomCluster(K, pd.Series(range(num_unique_students)))
    init_params = dict()
    init_params['Lambda'] = {'value': 0.1}
    init_params['epsilon'] = {'value': 0.1}
    init_params['cluster'] = {'value': true_cluster_structure}
    K = 2
    num_iter = 10000
    thin = 100

    chain = model.singleChainMLE([sim_data['sim_data']], K, init_params)