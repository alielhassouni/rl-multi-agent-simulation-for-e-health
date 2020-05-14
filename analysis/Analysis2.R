library(ggplot2)
library(dplyr)
#library(maps)
#library(ggmap)
library(mongolite)
library(lubridate)
library(gridExtra)
library(plotly)

# Connect to mongodb database and read collection
my_collection = mongo(
  collection = "agent_state", 
  db = "experiment11") 
nr_days = 1
# Dop a count on the number of collections
my_collection$count()

# Get all collections
res = my_collection$find()

# Get rewards
res$rewards[[25]]
period0 = 1
period1 = nr_days
period2 = period1 + nr_days
period3 = period2 + nr_days

# Function that calculates average reward per day and makes a plot
view_plot_Agent <- function(agentid){
  # Get all collections
  res = my_collection$find() 
  
  agent_data = as.numeric(unlist(res$rewards[[agentid]]))
  res1 = c()
  lb = 1
  ub = lb +23
  
  
  for(i in 1:floor(length(agent_data)/24)) {
    
    #print(lb)
    #print(ub)
    
    s = mean(agent_data[lb:ub]) 
    #print(s)
    res1 = c(res1, s)
    
    lb = lb + 24
    ub = lb +23
  }
  y = res1
  x = c(1:length(res1))
  data = data.frame(x, y)
  
  #print(data)
  a = list(rep(0, period1))
  b = list(rep(0, period1))
  c = list(rep(0, period1))
  
  d = c(a,b,c)
  data = data.frame(data, d)
  
  p = plot_ly(data, x = ~x, y = ~y, type = 'scatter', mode = 'lines', color = ~d)
  p
  return(y)
  
}

view_plot_Agent(10)


mat <- matrix(, nrow = floor(length(res$rewards[[1]])/24), ncol = length(res$agent_id))

for(column in 1:length(res$agent_id)){
  print(column)
  averages <- view_plot_Agent(column)
  mat[, column] <- averages
}

means_all_agents <- rowMeans(mat, na.rm = FALSE, dims = 1)

print(data)

mat <-  cbind(mat,means_all_agents)

d = c(1:floor(length(res$rewards[[1]])/24))
means_all_agents = data.frame(mat, d)

p = plot_ly(means_all_agents, x=~d, y = ~means_all_agents, type = 'scatter', mode = 'lines', color = ~means_all_agents) 
p










# Create a plot
d = c(1:length(res1))

means = ()
for(j in 1:length(res$agent_id) ) {
  
  agent_data = as.numeric(unlist(res$rewards[[j]]))
  res1 = c()
  lb = 1
  ub = lb +23
  
  for(i in 1:floor(length(agent_data)/24)) {
    
    #print(lb)
    #print(ub)
    
    s = mean(agent_data[lb:ub]) 
    #print(s)
    res1 = c(res1, s)
    
    lb = lb + 24
    ub = lb +23
  }
  y = res1
  
  if(j ==1 ){
    d = do.call(rbind, Map(data.frame, A=d, B=y))
    #d = c(d, y)
  } else {
    
    d[paste(j, "agent")] <- "NA"
    d$agent_id = y
  }
  
  
  # d$agent_id = y
  
  
}

result = as.data.frame(d)
