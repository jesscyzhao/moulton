source('function/working.mle.r')
sim.data = read.csv('data/python_sim_data_seed_1.csv')
true.cluster = read.csv('data/python_cluster_structure_seed_1.csv')

num.ind = 20
K = 2
Lambda = 0.1
epsilon = 0.1
# cluster.structure = true.cluster

chain = run.mle(sim.data, K, num.ind, epsilon, Lambda, rep(1/K, K), num.iter = 10000, thin=100)

