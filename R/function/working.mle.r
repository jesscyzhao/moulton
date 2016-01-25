propose.cluster = function(x)
{
    num.state =length(table(x))
    this.ind = sample(1:length(x),1,replace=FALSE)
    values = as.numeric(names(table(x)))
    new.x = x;
    new.x[this.ind] = sample(values,1,replace=FALSE)
        return(new.x)
}


propose.cluster.1 = function(x,K)
{
    values = which(table(x)!=1)-1
    this.ind = sample(which(x %in% values),1,replace=FALSE)
    new.x = x 
    new.x[this.ind] = sample((1:K)-1,1,replace=FALSE)
        return(new.x)
}

clust.vec = function(i,cluster,eps)
{
    vec = rep(0,length(cluster))
    vec[cluster==i]=1/sum(cluster==i)*(1-eps);
    vec[cluster!=i]=eps/sum(cluster!=i)
    return(vec)
    
}


create.obs.mat = function(num.states,cluster,eps){
    obs.mat = matrix(unlist(lapply(0:(num.states-1),clust.vec,cluster,eps)),ncol=length(cluster),byrow=TRUE)

    return(obs.mat)
}

## switch.rate rate
create.trans.mat = function(num.states,lambda){
    trans.mat = array(lambda/(num.states-1), dim=c(num.states,num.states))
    
    diag(trans.mat) = 1-lambda
    
    return(trans.mat)
}



forward.better = function(data.set,num.states,obs.mat,trans.mat,pi){
    num.data = length(data.set)
    
    alpha = array(NA,c(num.states,num.data))
    for (i in 1:num.states){
        alpha[i,1] = log(pi[i]*obs.mat[i,data.set[1]+1])
    }
    
    for(t in 1:(num.data-1))
    {
        for(j in 1:num.states)
        {
            entry = rep(NA,num.states);
            for(i in 1:num.states)
            {
                #			print(c(t,i,j))
                
                entry[i] = alpha[i,t] + log(trans.mat[i,j])
                
                
                #total = total + alpha[i,t]*trans.mat[i,j];
            }
            max.at = max(entry)
            sum.1 = sum(exp(entry-max.at))
            final = max.at+log(sum.1)
            alpha[j,t+1] = final+log(obs.mat[j,data.set[t+1]+1]);
        }
    }
    
    max.final = max(alpha[,num.data])
    # Original code is wrong. log(sum(exp())) not sum(exp())
    llk = max.final + log(sum(exp(alpha[,num.data]-max.final)))
    return(llk)
    #return(alpha)
}


sim.data = function(num.data,num.states,cluster,eps,lambda,pi)
{
    num.ind = length(cluster)
    if (length(table(cluster))!=num.states)
    {
        print("Full cluster structure not represented")
    }
    obs.mat   = create.obs.mat(num.states,cluster,eps);
    trans.mat = create.trans.mat(num.states,lambda);
    
    sim.state = rep(NA,num.data);
    sim.data  = rep(NA,num.data);
    
    sim.state[1] = sample(0:(num.states-1),1,replace = FALSE,pi)
    #print(sim.state[1])
    #print(0:(num.ind-1))
    #print(obs.mat[sim.state[1]+1,])
    
    sim.data[1]  = sample(0:(num.ind-1),1,replace=FALSE,obs.mat[sim.state[1]+1,])
    
    for(i in 2:num.data)
    {
        last.state = sim.state[i-1];
        
        ## create new state
        if (runif(1) < trans.mat[last.state+1,last.state+1])
        {	sim.state[i] = sim.state[i-1];	}
        else
        {
            #	print(sim.state[i-1])
            r.states = c(0:(num.states-1))[-c(sim.state[i-1]+1)]
            #print(r.states)
            probs = trans.mat[sim.state[i-1]+1,1:num.states][-c(sim.state[i-1]+1)]
            #print(probs)
            sim.state[i] = sample(r.states,1,replace=FALSE,probs);
            #print(sim.state)
        }
        
        sim.data[i] = sample(0:(num.ind-1),1,replace=FALSE,obs.mat[sim.state[i]+1,])
    }
    return(cbind(sim.data,sim.state))
}



run.mle = function(data.set,num.states,num.ind,eps,lambda,pi,num.iter,thin){

cluster = rep(c(0:(num.states-1)),num.ind)[1:num.ind]
obs.mat = create.obs.mat(num.states,cluster,eps)
trans.mat = create.trans.mat(num.states,lambda)
num.ind = length(cluster)

curr.llk = sum(unlist(lapply(data.set,forward.better,num.states,obs.mat,trans.mat,pi)))



chain = rep(list(),num.iter/thin)

for( i in 1:num.iter)
{
    
    test = runif(1);
    
    new.lambda = lambda;
    new.eps = eps;
    new.cluster = cluster;
    if( test < 0.3)
    {	new.eps = runif(1);	}
    else if (test < 0.6)
    {	new.lambda = runif(1);	}
    else
    {	# new.cluster = propose.cluster.1(cluster,num.states)	
      new.cluster = cluster	}
    
    curr.llk = sum(unlist(lapply(data.set,forward.better,num.states,obs.mat,trans.mat,pi)))
    
    new.obs.mat = create.obs.mat(num.states,new.cluster,new.eps)
    new.trans.mat = create.trans.mat(num.states,new.lambda)
    prop.llk = sum(unlist(lapply(data.set,forward.better,num.states,new.obs.mat,new.trans.mat,pi)))
    
    
    if(prop.llk > curr.llk-0.5)
    {
        eps = new.eps;
        lambda = new.lambda;
        cluster = new.cluster;
        obs.mat = new.obs.mat;
        trans.mat = new.trans.mat;
        curr.lkk = prop.llk;
    }
    
    if( i %% thin ==0)
    {
        chain[[i/thin]] = list(i,eps,lambda,cluster,curr.llk)
        print(c(chain[[i/thin]][[5]],chain[[i/thin]][[2]], chain[[i/thin]][[3]]))
    }
    
}

#print(chain[[num.iter/thin]][[4]])

return(chain)

}





KL.divergence = function(result,clust,num.states){
    compare.table = table(result,clust)
    ln.pq.dif  = matrix(rep(0,num.states^2),nrow = num.states)
    p.matrix = compare.table
    for (i in 1:dim(compare.table)[1]){
        p.matrix[i,] = compare.table[i,]/sum(compare.table[i,])
        zero = which(p.matrix[i,] ==0)
        ln.pq.dif[i,] = p.matrix[i,]
        ln.pq.dif[i,-zero] = p.matrix[i,-zero]*(log(p.matrix[i,-zero]/(1/num.states)))
    }
    sum = sum(ln.pq.dif)
    return(sum)
}
