#I have been practicing gradient descent with python, I want to try it with R today.
# The dataset I used is new york house.csv, but the cut version with only two variables--bedrooms and sqft, and price
df = read.csv('house.csv')
summary(df)
lm(df$price~df$bd+df$sqft)
#GD(df, alpha = 0.1, maxIter = 10, seed = NULL)
X = cbind(rep(1, 10000000), df)
head(X)
head(df)
X = subset(X, select = -c(price) )
head(y)

X$sqft = (X$sqft-mean(X$sqft))/sd(X$sqft)
X$bd = (X$bd-mean(X$bd))/sd(X$bd)
y = df[c('price')]
tail(y)
X = data.matrix(X)
y = data.matrix(y)
m = nrow(y)
theta<-rep(0,3)

compCost<-function(X, y, theta){
  m <- row(y)
  J <- sum((X%*%theta- y)^2)/(2*m)
  return(J)
}

gradDescent<-function(X, y, theta, alpha, num_iters){
  m <- nrow(y)
  J_hist <- rep(0, num_iters)
  for(i in 1:num_iters){
    
    # this is a vectorized form for the gradient of the cost function
    # X is a 10000000x3 matrix, theta is a 3x1 column vector, y is a 10000000x1 column vector
    # X transpose is a 3x10000000 matrix. So t(X)%*%(X%*%theta - y) is a 3x1 column vector
    theta <- theta - alpha*(1/m)*(t(X)%*%(X%*%theta - y))
    
    # this for-loop records the cost history for every iterative move of the gradient descent,
    # and it is obtained for plotting number of iterations against cost history.
    J_hist[i]  <- compCost(X, y, theta)
  }
  # for a R function to return two values, we need to use a list to store them:
  results<-list(theta, J_hist)
  return(results)
}

alpha <- 0.15
num_iters <- 100
results <- gradDescent(X, y, theta, alpha, num_iters)
theta <- results[[1]]
cost_hist <- results[[2]]
print(theta)
plot(cost_hist)

#compare the result with the one manually calculated
df_scale = df
df_scale$sqft = (df_scale$sqft-mean(df_scale$sqft))/sd(df_scale$sqft)/1e+05
df_scale$bd = (df_scale$bd-mean(df_scale$bd))/sd(df_scale$bd) / 1e+05
lm(df_scale$price~df_scale$bd+df_scale$sqft) 
summary(lmdf)


(solve(t(X)%*%X))%*%t(X)%*%y





        
