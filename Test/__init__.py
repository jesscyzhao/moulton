from HMM.Model import HMM

import pandas as pd

def main():
    model = HMM()
    K = 10
    student_list =pd.Series(range(20))
    random_cluster = model.createRandomCluster(K, student_list)

    update_cluster = model.updateRandomCluster(random_cluster)

    obs_sequence = pd.Series([1,1,1,2,2,2,3,3,3])

    state_sequence = model.convertObsToStateSequence(obs_sequence, update_cluster)

    error_matrix = model.getErrorRateMatrix(update_cluster, K, 0.1)

    error_dict = error_matrix.to_dict()

    print(pd.DataFrame({'error': error_matrix[1], 'cluster': update_cluster}))
if __name__ == '__main__':
    main()