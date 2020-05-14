library(ggplot2)
library(dplyr)
library(mongolite)
library(lubridate)
library(gridExtra)
library(plotly)
library(broom)

# Connect to mongodb database and read collection
Experiment = "experiment75"
my_collection = mongo(
  collection = "agent_state", 
  db = Experiment) 

rerun = F
nr_days = 3
# Dop a count on the number of collections
my_collection$count()

# Get all collections
res = my_collection$find()

# Get rewards
period0 = 1
period1 = 5
period2 = period1 + 1
period3 = period2 + 1

# Function that calculates average reward per day and makes a plot
view_plot_Agent <- function(agentid, start){
  agent_data = as.numeric(unlist(res$rewards[[agentid]]))
  res1 = c()
  lb = 1
  ub = lb +23

  for(i in start:floor(length(agent_data)/24)) {
    s = mean(agent_data[lb:ub]) 
    res1 = c(res1, s)
    lb = lb + 24
    ub = lb +23
  }

  y = res1
  x = c(1:length(res1))
  data = data.frame(x, y)

  a = list(rep(0, period1))
  b = list(rep(0, period2))
  c = list(rep(0, period3))
  d = c(a,b,c)
  data = data.frame(data, c(1:nrow(data)))
  p = plot_ly(data, x = ~x, y = ~y, type = 'scatter', mode = 'lines', color = ~d)
  p
  return(y)
}

view_plot_Agent_2 <- function(agentid, start){
  agent_data = as.numeric(unlist(res$rewards[[agentid]]))
  res1 = c()
  lb = 1
  ub = lb +23

  for(i in start:floor(length(agent_data)/24)) {
    s = mean(agent_data[lb:ub]) 
    res1 = c(res1, s)
    lb = lb + 24
    ub = lb +23
  }

  y = res1
  x = c(1:length(res1))
  data = data.frame(x, y)

  a = list(rep(0, period1))
  b = list(rep(0, period2))
  c = list(rep(0, period3))
  
  d = c(a,b,c)
  data = data.frame(data, c(1:nrow(data)))
  p = plot_ly(data, x = ~x, y = ~y, type = 'scatter', mode = 'lines', color = ~d)
  p
}

view_plot_Agent_2(11,1)

if(!rerun) {
  mat <- matrix(, nrow = floor(length(res$rewards[[1]])/24), ncol = length(res$agent_id))
  for(column in 1:length(res$agent_id)){
    print(column)
    averages <- view_plot_Agent(column, 1)
    mat[, column] <- averages
  }
} else {
  mat_new <- matrix(, nrow = floor(length(res$rewards[[1]])/24 - nrow(mat)), ncol = length(res$agent_id))
  for(column in 1:length(res$agent_id)){
    print(column)
    averages <- view_plot_Agent(column, nrow(mat)+1)
    mat_new[, column] <- averages
  }
  mat <- rbind(mat, mat_new)
}

means_all_agents <- rowMeans(mat, na.rm = FALSE, dims = 1)
d = c(1:floor(length(res$rewards[[1]])/24))
means_all_agents = data.frame(means_all_agents, d)
means_all_agents$x = means_all_agents$d

m <- loess(means_all_agents ~ d, data = means_all_agents)
p = plot_ly(means_all_agents, x = ~d, y = ~means_all_agents, type = 'scatter', mode = 'lines', color = ~means_all_agents) %>%
  add_lines(y = ~fitted(loess(means_all_agents ~ d)), line = list(color = 'rgba(7, 164, 181, 1)'), name = "Loess Smoother") %>%
  add_trace(x = ~d, y = ~means_all_agents, mode = 'markers') %>%
  layout(
    title = Experiment)%>%
  add_ribbons(data = augment(m),
              ymin = ~.fitted - 1.96 * .se.fit,
              ymax = ~.fitted + 1.96 * .se.fit,
              line = list(color = 'rgba(7, 164, 181, 0.05)'),
              fillcolor = 'rgba(7, 164, 181, 0.2)',
              name = "Standard Error")
p
                      