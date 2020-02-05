# Domain Adaptation: comparing different models performance in 3 Source-Target tasks
# Air temperature data from France
# Included models: DNN, XGBoost, LMM, GMRF
# Validation approach: repeated train-test sampling according to source and target distributions:
# In all tasks training and test sets are built as follow: 
#   1. spatial locations are randomly drawn from P_T(g)
#   2. all the temporal data related to these locations serve for testing
#   3. subset from the remaining data serve for training
# In task 1 P_T(g) implies uniform distribution...


library(Matrix)
library(ramps)
library(foreach)
library(doMC)
library(MASS)
library(scales)
library(xgboost)
library(caret)
library(data.table)
library(magrittr)
library(ggplot2)
library(gridExtra)
library(magrittr)
library(leaflet)
library(sp)
library(raster)
library(keras)
library(lme4)
library(glmnet)
library(randomForest)
library(spatstat)
library(sf)
library(mvnfast)
library(LMERConvenienceFunctions)
library(INLA)
library(rhdf5)

rm(list = ls())
gc()

# some training parameters
trn_ratio <- 0.75
tst_num_stns <- 600

# source density estimation and IW
grid_n <- 500
sigma_density <- 0.5

# IW
IWadj_ratio <- 0.95

# dnn
epochs <- 40
batch_size <- 64
val_ratio <- 0.3

# xgboost
num_round <- 3000 
eta <- 0.01
max_depth <- 6
xgb_earlys <- 10

# general
verbose <- 0

# validation
CV_k <- 30


# raw data
db <- readRDS("~/DA/data/db_france.Rda")
coords <- unique(db[,.(lon,lat,stn_id)])
double <- coords[,by=.(lon,lat) ,.N][N>1]
coords <- coords[!(lon %in% double$lon & lat %in% double$lat), ]


borders <- getData('GADM', country='FRA', level=0)
border_coords <- borders@polygons[[1]]@Polygons[[350]]@coords %>% data.table()
pol <- st_polygon(list(as.matrix(border_coords)))
pbuf <- st_buffer(pol, .3) %>% as.matrix()
ow <- owin(poly = pbuf[nrow(pbuf):1,])

features <- c("date",
              "lon_s","lat_s","lon_s_2","lat_s_2","lon_s_sin","lat_s_sin","lon_s_cos","lat_s_cos",
              "climate_type","reg",
              "aqua_night_lst","aqua_emis","aqua_ndvi","elev","pop","clc_artificial",
              "clc_water","clc_bare","clc_vegetation")

# functions
weightedloss <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  weights <- getinfo(dtrain, "weight")
  grad <- weights * (preds - labels)
  hess <- weights
  return(list(grad = grad, hess = hess))
}
evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  weights <- getinfo(dtrain, "weight")
  err <- sum(weights/sum(weights) * (labels - preds)^2)
  return(list(metric = "werror", value = err))
}

build_model <- function() {
  model <- keras_model_sequential() %>%
    layer_dense(units = 1024, activation = "tanh",
                input_shape = dim(train_input)[2] ) %>% #_trn
    layer_dense(units = 512, activation = "tanh") %>%
    layer_dense(units = 128, activation = "tanh") %>%
    layer_dense(units = 1)
  model %>% compile(
    loss = "mean_squared_error",
    optimizer = optimizer_sgd(lr = 0.001, momentum = 0.9),
    metrics = list("mse","mae")
  )
  model
}






# Task 1
task1_results <- matrix(NA, CV_k, 6)
colnames(task1_results) <- c("mse_dnn", "mse_wdnn", "mse_xgboost", "mse_wxgboost", "mse_lmm", "mse_wlmm")

for (j in 1:CV_k) {
  
  # generating train and test
  sample_P_T <- spsample(borders, 600, type = "regular")
  intest <- integer(nrow(sample_P_T@coords))
  for (i in 1:nrow(sample_P_T@coords)) {
    di <- (sample_P_T@coords[i,1]-coords$lon)^2 + (sample_P_T@coords[i,2]-coords$lat)^2 
    intest[i] <- coords$stn_id[which.min(di)]
  }
  intest <- unique(intest)
  coords$tst <- (coords$stn_id %in% intest)*1
  
  sample_P_S <- sample(coords[!(stn_id %in% intest), stn_id], nrow(coords[!(stn_id %in% intest),])*trn_ratio)
  coords$trn <- (coords$stn_id %in% sample_P_S)*1
  
  coords_trn <- coords[trn==1,]; coords_tst <- coords[tst==1,]
  stn_trn <- coords_trn$stn_id ; stn_tst <- coords_tst$stn_id
  db_trn <- db[stn_id %in% stn_trn,]
  db_tst <- db[stn_id %in% stn_tst,]
  
  # estimate source spatial dist., define target spatial dist., derive IW
  db_pppp <- ppp(x=coords_trn$lon, y=coords_trn$lat, window = ow)
  ker <- density.ppp(db_pppp, edge = T, dimyx = c(grid_n,grid_n), sigma = sigma_density)
  all_f_S <- ker$v / sum(ker$v, na.rm = T)
  xindex <- sapply(coords_trn$lon, function(s) which.min(na.omit(abs(ker$xcol-s))))
  yindex <- sapply(coords_trn$lat, function(s) which.min(na.omit(abs(ker$yrow-s))))
  f_S <- numeric(length(coords_trn$lon))
  for (i in seq_along(f_S)) {f_S[i] <- all_f_S[yindex[i],xindex[i]]}
  rownames(all_f_S) <- ker$xcol
  colnames(all_f_S) <- ker$yrow
  
  all_P_T <- matrix(1/grid_n^2, grid_n, grid_n)
  rownames(all_P_T) <- ker$xcol
  colnames(all_P_T) <- ker$yrow
  P_T <- numeric(nrow(coords_trn))
  for (i in seq_along(P_T)) {P_T[i] <- all_P_T[xindex[i],yindex[i]]}
  
  all_PT_fS <- all_P_T/all_f_S
  coords_trn$f_S <- f_S
  coords_trn$P_T <- P_T
  coords_trn$IW <- P_T/f_S
  coords_trn$IWadj <- ifelse(coords_trn$IW > quantile(P_T/f_S, IWadj_ratio, na.rm = T),
                              quantile(P_T/f_S, IWadj_ratio, na.rm = T), coords_trn$IW)
  db_trn <- merge(x=db_trn, y=coords_trn[,.(stn_id,IWadj)], by="stn_id")
  
  # prepare data for learning
  shuffled <- sample(1:nrow(db_trn))
  db_trn <- db_trn[shuffled,]
  train_input <- sparse.model.matrix(data = db_trn[,..features ], ~.) 
  train_output <- db_trn$tmin
  train_weights <- db_trn$IWadj
  test_input <- sparse.model.matrix(data = db_tst[,..features], ~. ) 
  test_output <- db_tst$tmin
  
  m_dnn1 <- build_model()
  history1 <- m_dnn1 %>% fit(
    train_input, 
    train_output,
    batch_size = batch_size,
    epochs = 50,
    verbose = verbose )
  
  m_wdnn1 <- build_model()
  history2 <- m_wdnn1 %>% fit(
    train_input, 
    train_output,
    batch_size = batch_size,
    epochs = 50,
    sample_weight = train_weights,
    verbose = verbose )
  
  p_dnn1 <- m_dnn1 %>% predict(test_input)
  p_wdnn1 <- m_wdnn1 %>% predict(test_input)
  (mse_dnn1 <- mean((p_dnn1 - db_tst$tmin)^2) )
  (mse_wdnn1 <- mean((p_wdnn1 - db_tst$tmin)^2) )
  
  # xgboost
  dtrain <- xgb.DMatrix(train_input, label = train_output)
  dtrain_w <- xgb.DMatrix(train_input, label = train_output, weight = train_weights)
  dtest <- xgb.DMatrix(test_input, label = test_output)
  watchlist <- list(eval = dtest, train = dtrain)
  watchlistw <- list(eval = dtest, train = dtrain_w)
  param <- list(max_depth=6, eta=0.2, verbosity=1)
  paramw <- list(max_depth=6, eta=0.2, verbosity=1, objective=weightedloss, eval_metric=evalerror)
  
  m_xgboost1 <- xgb.train(param, dtrain, num_round, watchlist, verbose = verbose,
                          earlystoppingrounds=xgb_earlys)
  m_wxgboost1 <- xgb.train(paramw, dtrain_w, num_round, watchlistw, verbose = verbose,
                           earlystoppingrounds=xgb_earlys)
  
  p_xgboost1 <- predict(m_xgboost1, dtest)
  p_wxgboost1 <- predict(m_wxgboost1, dtest)
  (mse_xgboost1 <- mean((p_xgboost1 - db_tst$tmin)^2))
  (mse_wxgboost1 <- mean((p_wxgboost1 - db_tst$tmin)^2))
  
  # lmm
  m_lmm1 <- lmer(tmin ~
                   lon_s+lat_s+lon_s_2+lat_s_2+lon_s_sin+lat_s_sin+lon_s_cos+lat_s_cos+
                   stn_elev+aqua_night_lst+aqua_emis+aqua_ndvi+elev+pop+clc_artificial+
                   clc_water+clc_bare+clc_vegetation+climate_type+
                   (1+aqua_night_lst|date/reg), db_trn )
  
  m_wlmm1 <- lmer(tmin ~
                    lon_s+lat_s+lon_s_2+lat_s_2+lon_s_sin+lat_s_sin+lon_s_cos+lat_s_cos+
                    stn_elev+aqua_night_lst+aqua_emis+aqua_ndvi+elev+pop+clc_artificial+
                    clc_water+clc_bare+clc_vegetation+climate_type+
                    (1+aqua_night_lst|date/reg), db_trn, weights = db_trn$IWadj )
  
  p_lmm1 <- predict(m_lmm1, db_tst, allow.new.levels=TRUE, re.form=NULL)
  p_wlmm1 <- predict(m_wlmm1, db_tst, allow.new.levels=TRUE, re.form=NULL)
  (mse_lmm1 <- mean((p_lmm1 - db_tst$tmin)^2))
  (mse_wlmm1 <- mean((p_wlmm1 - db_tst$tmin)^2))
  
  (task1_results[j,] <- c(mse_dnn1, mse_wdnn1, 
                          mse_xgboost1, mse_wxgboost1,
                          mse_lmm1, mse_wlmm1))
  
  print(paste("task 1: ", j))
  print(task1_results[j,])
}

write.csv(task1_results, "~/DA/data/task1_results.csv")



# Task 2
base <- read.csv("~/DA/data/popfr19752010.csv", header = T) %>% data.table()
bydep <- base[,.(pop = sum(pop_2010)), by = "dep"]
basem <- merge(bydep,base)[,.(long,lat,pop)]
pop <- numeric(nrow(coords))
for (i in 1:nrow(coords)) {
  pop[i] <- basem$pop[which.min( (coords$lon[i]-basem$long)^2 + (coords$lat[i]-basem$lat)^2)]
}
coords$pop <- (pop/1e4)^2

task2_results <- matrix(NA, CV_k, 6)
colnames(task2_results) <- c("mse_dnn", "mse_wdnn","mse_xgboost", "mse_wxgboost","mse_lmm", "mse_wlmm")

for (j in 1:CV_k) {

  sample_P_T <- sample(rep(coords$stn_id, coords$pop ), size = 600, replace = F)
  coords$tst <- (coords$stn_id %in% sample_P_T)*1
  
  sample_P_S <- spsample(borders, 1500, type = "regular")
  intrain <- integer(nrow(sample_P_S@coords))
  for (i in 1:nrow(sample_P_S@coords)) {
    di <- (sample_P_S@coords[i,1]-coords[tst!=1,lon])^2 + (sample_P_S@coords[i,2]-coords[tst!=1,lat])^2
    intrain[i] <- coords[tst!=1,stn_id][which.min(di)]
  }
  intrain <- unique(intrain)
  coords$trn <- (coords$stn_id %in% intrain)*1
  
  coords_trn <- coords[trn==1,]; coords_tst <- coords[tst==1,]
  stn_trn <- coords_trn$stn_id ; stn_tst <- coords_tst$stn_id
  db_trn <- db[stn_id %in% stn_trn,]
  db_tst <- db[stn_id %in% stn_tst,]

  coords_trn$f_S <- 1/grid_n^2
  coords_trn$P_T <- coords_trn$pop %>% scales::rescale(to=c(0.5/grid_n^2, 2/grid_n^2))
  coords_trn$IW <- coords_trn$P_T/coords_trn$f_S
  coords_trn$IWadj <- ifelse(coords_trn$IW > quantile(coords_trn$IW, IWadj_ratio, na.rm = T),
                             quantile(coords_trn$IW, IWadj_ratio, na.rm = T), coords_trn$IW)
  coords_trn$IWadj <- ifelse(coords_trn$IW > 1, 1, coords_trn$IW)
  
  db_trn <- merge(x=db_trn, y=coords_trn[,.(stn_id,IWadj)], by="stn_id")
  shuffled <- sample(1:nrow(db_trn))
  db_trn <- db_trn[shuffled,]
  train_input <- sparse.model.matrix(data = db_trn[,..features ], ~.)
  train_output <- db_trn$tmin
  train_weights <- db_trn$IWadj
  test_input <- sparse.model.matrix(data = db_tst[,..features], ~. )
  test_output <- db_tst$tmin
  
  
  # dnn
  m_dnn2 <- build_model()
  history1 <- m_dnn2 %>% fit(
    train_input,
    train_output,
    batch_size = batch_size,
    epochs = 35,
    verbose = verbose )
  
  m_wdnn2 <- build_model()
  history2 <- m_wdnn2 %>% fit(
    train_input,
    train_output,
    batch_size = batch_size,
    epochs = 35,
    sample_weight = train_weights,
    verbose = verbose )
  
  p_dnn2 <- m_dnn2 %>% predict(test_input)
  p_wdnn2 <- m_wdnn2 %>% predict(test_input)
  mse_dnn2 <- mean((p_dnn2 - db_tst$tmin)^2)
  mse_wdnn2 <- mean((p_wdnn2 - db_tst$tmin)^2)
  
  # xgboost
  dtrain <- xgb.DMatrix(train_input, label = train_output)
  dtrain_w <- xgb.DMatrix(train_input, label = train_output, weight = train_weights)
  dtest <- xgb.DMatrix(test_input, label = test_output)
  watchlist <- list(eval = dtest, train = dtrain)
  watchlistw <- list(eval = dtest, train = dtrain_w)
  param <- list(max_depth=6, eta=0.2, verbosity=1)
  paramw <- list(max_depth=6, eta=0.2, verbosity=1, objective=weightedloss, eval_metric=evalerror)
  
  m_xgboost2 <- xgb.train(param, dtrain, num_round, watchlist, verbose = verbose)
  m_wxgboost2 <- xgb.train(paramw, dtrain_w, num_round, watchlistw, verbose = verbose)
  
  p_xgboost2 <- predict(m_xgboost2, dtest)
  p_wxgboost2 <- predict(m_wxgboost2, dtest)
  mse_xgboost2 <- mean((p_xgboost2 - db_tst$tmin)^2)
  mse_wxgboost2 <- mean((p_wxgboost2 - db_tst$tmin)^2)
  
  # lmm
  m_lmm2 <- lmer(tmin ~
                   lon_s+lat_s+lon_s_2+lat_s_2+lon_s_sin+lat_s_sin+lon_s_cos+lat_s_cos+
                   stn_elev+aqua_night_lst+aqua_emis+aqua_ndvi+elev+pop+clc_artificial+
                   clc_water+clc_bare+clc_vegetation+climate_type+
                   (1+aqua_night_lst|date/reg), db_trn)
  m_wlmm2 <-lmer(tmin ~
                   lon_s+lat_s+lon_s_2+lat_s_2+lon_s_sin+lat_s_sin+lon_s_cos+lat_s_cos+
                   stn_elev+aqua_night_lst+aqua_emis+aqua_ndvi+elev+pop+clc_artificial+
                   clc_water+clc_bare+clc_vegetation+climate_type+
                   (1+aqua_night_lst|date/reg), db_trn, weights = train_weights)
  
  p_lmm2 <- predict(m_lmm2, db_tst, allow.new.levels=TRUE, re.form=NULL)
  p_wlmm2 <- predict(m_wlmm2, db_tst, allow.new.levels=TRUE, re.form=NULL)
  mse_lmm2 <- mean((p_lmm2 - db_tst$tmin)^2)
  mse_wlmm2 <- mean((p_wlmm2 - db_tst$tmin)^2)
  
  task2_results[j,] <- c(mse_dnn2, mse_wdnn2, mse_xgboost2, mse_wxgboost2, mse_lmm2, mse_wlmm2)
  
  print(paste("task 2: ", j))
  print(task2_results[j,])
}

write.csv(task2_results, "~/DA/data/task2_results.csv")







# Task 3
task3_results <- matrix(NA, CV_k, 6)
colnames(task3_results) <- c("mse_dnn", "mse_wdnn", "mse_xgboost", "mse_wxgboost", "mse_lmm", "mse_wlmm")
Q_matern = corRMatern(value = c(10,2), form = ~ lon + lat)
cor_mat <- nlme::corMatrix(Initialize(Q_matern,coords[,.(lon,lat)]))
set.seed(254); coords$Q0matern <- (rmvn(1, mu=rep(1,nrow(coords)),sigma = cor_mat) %>% scales::rescale(to = c(0,1)) )^2

for (j in 1:CV_k) {
  # generating train and test
  sample_P_T <- sample(rep(coords$stn_id, 100*coords$Q0matern), size = tst_num_stns, replace = F)
  coords$tst <- (coords$stn_id %in% sample_P_T)*1
  sample_P_S <- sample(coords[tst != 1, stn_id], nrow(coords[tst != 1,])*trn_ratio)
  coords$trn <- (coords$stn_id %in% sample_P_S)*1
  coords_trn <- coords[trn==1,]; coords_tst <- coords[tst==1,]
  stn_trn <- coords_trn$stn_id; stn_tst <- coords_tst$stn_id
  db_trn <- db[stn_id %in% stn_trn,]
  db_tst <- db[stn_id %in% stn_tst,]
  
  # estimate f_S, define P_T, derive IW
  db_pppp <- ppp(x=coords_trn$lon, y=coords_trn$lat, window = ow)
  ker <- density.ppp(db_pppp, edge = T, dimyx = c(grid_n,grid_n), sigma = 0.75)
  all_f_S <- ker$v / sum(ker$v, na.rm = T)
  xindex <- sapply(coords_trn$lon, function(s) which.min(na.omit(abs(ker$xcol-s))))
  yindex <- sapply(coords_trn$lat, function(s) which.min(na.omit(abs(ker$yrow-s))))
  f_S <- numeric(length(coords_trn$lon))
  for (i in seq_along(f_S)) {f_S[i] <- all_f_S[yindex[i],xindex[i]]}
  rownames(all_f_S) <- ker$xcol
  colnames(all_f_S) <- ker$yrow
  
  coords_trn$f_S <- f_S
  coords_trn$P_T <- coords_trn$Q0matern  %>% scales::rescale(to=range(f_S))
  coords_trn$IW <- coords_trn$P_T/coords_trn$f_S
  coords_trn$IWadj <- ifelse(coords_trn$IW > quantile(coords_trn$IW, IWadj_ratio),
                              quantile(coords_trn$IW, IWadj_ratio), coords_trn$IW)
  db_trn <- merge(x=db_trn, y=coords_trn[,.(stn_id,IWadj)], by="stn_id")
  
  shuffled <- sample(1:nrow(db_trn))
  db_trn <- db_trn[shuffled,]
  train_input <- sparse.model.matrix(data = db_trn[,..features ], ~.) # + aqua_night_lst*date*climate_type)
  train_output <- db_trn$tmin
  train_weights <- db_trn$IWadj#^lambda
  test_input <- sparse.model.matrix(data = db_tst[,..features], ~. ) # + aqua_night_lst*date*climate_type)
  test_output <- db_tst$tmin

  # dnn
  m_dnn3 <- build_model()
  history1 <- m_dnn3 %>% fit(
    train_input,
    train_output,
    batch_size = batch_size,
    epochs = 50, 
    verbose = verbose )
  
  m_wdnn3 <- build_model()
  history3 <- m_wdnn3 %>% fit(
    train_input,
    train_output,
    batch_size = batch_size,
    epochs = 90, 
    sample_weight = train_weights,
    verbose = verbose )
  
  p_dnn3 <- m_dnn3 %>% predict(test_input)
  p_wdnn3 <- m_wdnn3 %>% predict(test_input)
  mse_dnn3 <- mean((p_dnn3 - db_tst$tmin)^2)
  mse_wdnn3 <- mean((p_wdnn3 - db_tst$tmin)^2)
  
  # xgboost
  dtrain <- xgb.DMatrix(train_input, label = train_output)
  dtrain_w <- xgb.DMatrix(train_input, label = train_output, weight = train_weights)
  dtest <- xgb.DMatrix(test_input, label = test_output)
  watchlist <- list(eval = dtest, train = dtrain)
  watchlistw <- list(eval = dtest, train = dtrain_w)
  param <- list(max_depth=max_depth, eta=0.2, verbosity=1)
  paramw <- list(max_depth=max_depth, eta=0.2, verbosity=1, objective=weightedloss, eval_metric=evalerror)
  
  m_xgboost3 <- xgb.train(param, dtrain, 3000, watchlist, verbose = 0)
  m_wxgboost3 <- xgb.train(paramw, dtrain_w, 3000, watchlistw, verbose = 0)
  
  p_xgboost3 <- predict(m_xgboost3, dtest)
  p_wxgboost3 <- predict(m_wxgboost3, dtest)
  mse_xgboost3 <- mean((p_xgboost3 - db_tst$tmin)^2)
  mse_wxgboost3 <- mean((p_wxgboost3 - db_tst$tmin)^2)
  
  # lmm
  m_lmm3 <- lmer(tmin ~
                   lon_s+lat_s+lon_s_2+lat_s_2+lon_s_sin+lat_s_sin+lon_s_cos+lat_s_cos+
                   stn_elev+aqua_night_lst+aqua_emis+aqua_ndvi+elev+pop+clc_artificial+
                   clc_water+clc_bare+clc_vegetation+climate_type+
                   (1+aqua_night_lst|date/reg), db_trn) #)
  m_wlmm3 <-  lmer(tmin ~
                     lon_s+lat_s+lon_s_2+lat_s_2+lon_s_sin+lat_s_sin+lon_s_cos+lat_s_cos+
                     stn_elev+aqua_night_lst+aqua_emis+aqua_ndvi+elev+pop+clc_artificial+
                     clc_water+clc_bare+clc_vegetation+climate_type+
                     (1+aqua_night_lst|date/reg), db_trn, weights = db_trn$IWadj) #)
  
  p_lmm3 <- predict(m_lmm3, db_tst, allow.new.levels=TRUE, re.form=NULL)
  p_wlmm3 <- predict(m_wlmm3, db_tst, allow.new.levels=TRUE, re.form=NULL)
  mse_lmm3 <- mean((p_lmm3 - db_tst$tmin)^2)
  mse_wlmm3 <- mean((p_wlmm3 - db_tst$tmin)^2)

  task3_results[j,] <- c(mse_dnn3, mse_wdnn3, mse_xgboost3, mse_wxgboost3, mse_lmm3, mse_wlmm3)
  
  print(paste("task 3: ", j))
  print(task3_results[j,])
}

write.csv(task3_results, "~/DA/data/task3_results.csv")



r1 <- read.csv("~/DA/data/task1_results.csv")
r2 <- read.csv("~/DA/data/task2_results.csv")
r3 <- read.csv("~/DA/data/task3_results.csv")

alltasks <- cbind(rbind(r1[,-1],r2[,-1],r3[,-1]),
                  task = rep(c("Task 1: Uniform Target",
                               "Task 2: Population based Target, Uniform Source",
                               "Task 3: Mostly West Target"), each = nrow(r1))) %>% data.table()
allm <- melt(alltasks, id.vars = "task", variable.name = "la")
alls <- allm[, .("mean" = mean(value), "se" = sd(value)/sqrt(length(value))),
             by = .(task,la)][order(task,la),]
alls$la <- rep(c("DNN", "DNN \n + IW",
                 "XGB", "XGB \n + IW",
                 "LMM", "LMM \n + IW"), 3)
alls$iw <- rep(c("no iw","iw"),3*3)
alls$model <- rep(c("dnn","dnn","xgb","xgb","lmm","lmm"),3) 

ggplot(alls, aes(x=la, y=mean, color = model)) +
  facet_wrap(~task, scales = "free") +
  geom_errorbar(aes(ymin=mean-se, ymax=mean+se), width=.2, size = 1) +
  geom_point(size = 3, shape = 19, stroke = 2) +
  geom_point(data = alls[iw == "iw",], aes(x=la,y=mean), size = 1, shape = 19) +
  theme_bw() + 
  theme(legend.position = "none", strip.text = element_text(size=12), 
        text = element_text(size=12),
        axis.text.x = element_text(size = 12)) + 
  labs(x = "\n Learning Algorithm", y = "Cross-Validated MSE") 

ggsave("~/DA/charts/results.png")
