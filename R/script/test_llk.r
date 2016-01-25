source('function/working.mle.r')
print('Test functions built up to llk. Compare the result in R and Python.')


K=3 
Lambda=0.1 
epsilon=0.1
obs_seq = c(0,1,2,3,4)
init_cluster = c(1,2,1,0,1)

error_rate_matrix = create.obs.mat(K, init_cluster, epsilon)
switch_rate_matrix = create.trans.mat(K, Lambda)

print(error_rate_matrix)
print(switch_rate_matrix)

forward_llk = forward.better(obs_seq, K, error_rate_matrix, switch_rate_matrix, rep(1/K, K))

print(forward_llk)

print('Conclusion: the forward.better function is wrong. After fixing it, the result is the same as the python result.')
