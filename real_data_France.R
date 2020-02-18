# Domain Adaptation: comparing different models performance in 3 Source-Target tasks
# Air temperature data from France
# Considered models: Deep Neural Network (DNN), Gradient Boosting trees (XGB), Linear Mixed Model (LMM)
# Learning algorithm: least squares minimization (naive), and importance-weighted least-square (IW) for domain adaptation
# In total: 6 predictors - DNN, IWDNN, XGB, IWXGB, LMM, IWLMM
# Validation approach: repeated train-test sampling according to source and target distributions:
# For each task, we compare predictors’ accuracy by repeatingthe following procedure 50 times, then averaging the results
#   1. spatial locations are randomly drawn from P_T(g)
#       In task 1 P_T(g) implies (spatial) Uniform distribution
#       In task 2 P_T(g) implies Population based distribution
#       In task 3 P_T(g) implies Northwest intense distribution
#      all the temporal data related to these locations serve for testing
#   2. 75% from the remaining data serve for training
#   3. predictors are fitted on the training 
#   4. For each of these 6 predictors, the target risk is the average loss on the test data.



library(xgboost)
library(data.table)
library(ggplot2)
library(gridExtra)
library(magrittr)
library(leaflet)
library(sp)
library(keras)
library(lme4)
library(spatstat)

rm(list = ls())
gc()

verbose <- 1

grid_n <- 500

CV_k <-50


# raw data
db <- readRDS("~/DA/data/db_france.Rda")
coords <- unique(db[,.(lon,lat,stn_id)])
double <- coords[,by=.(lon,lat) ,.N][N>1]
coords <- coords[!(lon %in% double$lon & lat %in% double$lat), ]


borders <- raster::getData('GADM', country='FRA', level=0)
ow <- readRDS("~/DA/data/ow.Rda")

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






##### Task 1 #####

task1_results <- matrix(NA, CV_k, 6)
colnames(task1_results) <- c("mse_dnn", "mse_wdnn", "mse_xgboost", "mse_wxgboost", "mse_lmm", "mse_wlmm")

for (j in 1:CV_k) {
  
  # sample 600 stations fron uniform stratified sampling for test
  sample_P_T <- sp::spsample(borders, 600, type = "stratified")
  intest <- integer(nrow(sample_P_T@coords))
  for (i in 1:nrow(sample_P_T@coords)) {
    di <- sqrt((sample_P_T@coords[i,1]-coords$lon)^2 + (sample_P_T@coords[i,2]-coords$lat)^2)
    intest[i] <- coords$stn_id[which.min(di)]
  }
  intest <- unique(intest)
  coords$tst <- (coords$stn_id %in% intest)*1
  
  # sample 75% from the remaining stations for training
  sample_P_S <- sample(coords[!(stn_id %in% intest), stn_id], nrow(coords[!(stn_id %in% intest),])*0.75)
  coords$trn <- (coords$stn_id %in% sample_P_S)*1
  
  # generating train and test
  coords_trn <- coords[trn==1,]; coords_tst <- coords[tst==1,]
  stn_trn <- coords_trn$stn_id ; stn_tst <- coords_tst$stn_id
  db_trn <- db[stn_id %in% stn_trn,]
  db_tst <- db[stn_id %in% stn_tst,]
  
  # estimate f_S
  db_pppp <- ppp(x=coords_trn$lon, y=coords_trn$lat, window = ow)
  ker <- density.ppp(db_pppp, edge = T, dimyx = c(grid_n,grid_n), sigma = 0.5)
  all_f_S <- ker$v / sum(ker$v, na.rm = T)
  xindex <- sapply(coords_trn$lon, function(s) which.min(na.omit(abs(ker$xcol-s))))
  yindex <- sapply(coords_trn$lat, function(s) which.min(na.omit(abs(ker$yrow-s))))
  f_S <- numeric(length(coords_trn$lon))
  for (i in seq_along(f_S)) {f_S[i] <- all_f_S[yindex[i],xindex[i]]}
  rownames(all_f_S) <- ker$xcol
  colnames(all_f_S) <- ker$yrow
  
  # define uniform P_T
  all_P_T <- matrix(1/sum(!is.na(all_f_S)),  grid_n, grid_n)
  rownames(all_P_T) <- ker$xcol
  colnames(all_P_T) <- ker$yrow
  P_T <- numeric(nrow(coords_trn))
  for (i in seq_along(P_T)) {P_T[i] <- all_P_T[xindex[i],yindex[i]]}
  
  # derive IW
  all_PT_fS <- all_P_T/all_f_S
  coords_trn$f_S <- f_S
  coords_trn$P_T <- P_T
  coords_trn$IW <- P_T/f_S
  coords_trn$IWadj <- ifelse(coords_trn$IW > 1, 1, coords_trn$IW)
  db_trn <- merge(x=db_trn, y=coords_trn[,.(stn_id,IWadj)], by="stn_id")
  
  # prepare data for learning
  shuffled <- sample(1:nrow(db_trn))
  db_trn <- db_trn[shuffled,]
  train_input <- sparse.model.matrix(data = db_trn[,..features ], ~.) 
  train_output <- db_trn$tmin
  train_weights <- db_trn$IWadj
  test_input <- sparse.model.matrix(data = db_tst[,..features], ~. ) 
  test_output <- db_tst$tmin
  
  # models fitting
  # dnn
  m_dnn1 <- build_model()
  history1 <- m_dnn1 %>% fit(
    train_input, 
    train_output,
    batch_size = 64,
    epochs = 30,
    validation_data = list(test_input, test_output),
    verbose = verbose)
  m_wdnn1 <- build_model()
  history2 <- m_wdnn1 %>% fit(
    train_input, 
    train_output,
    batch_size = 64,
    epochs = 30,
    validation_data = list(test_input, test_output),
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
  param <- list(max_depth=6, eta=0.2, verbosity=verbose)
  paramw <- list(max_depth=6, eta=0.2, verbosity=verbose, objective=weightedloss, eval_metric=evalerror)
  m_xgboost1 <- xgb.train(param, dtrain, 2000, watchlist, verbose = verbose)
  m_wxgboost1 <- xgb.train(paramw, dtrain_w, 2000, watchlistw, verbose = verbose)
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
  
  # results
  (task1_results[j,] <- c(mse_dnn1, mse_wdnn1, mse_xgboost1, mse_wxgboost1, mse_lmm1, mse_wlmm1))
  
  print(paste("task 1: ", j))
  print(task1_results[j,])
}

write.csv(task1_results, "~/DA/data/task1_results.csv")






##### Task 2 #####

task2_results <- matrix(NA, CV_k, 6)
colnames(task2_results) <- c("mse_dnn", "mse_wdnn","mse_xgboost", "mse_wxgboost","mse_lmm", "mse_wlmm")


# load population based P_T and assign P_T(g)
all_P_T <- readRDS( "~/DA/data/task1_all_P_T.Rda")
P_T_coords <- numeric(nrow(coords))
xindex <- sapply(coords$lon, function(s) which.min(abs(as.numeric(colnames(all_P_T))-s)))
yindex <- sapply(coords$lat, function(s) which.min(abs(as.numeric(rownames(all_P_T))-s)))
for (i in seq_along(P_T_coords)) {
  P_T_coords[i] <- all_P_T[yindex[i],xindex[i]]
}

for (j in 1:CV_k) {
  # sample approximately 600 stations from P_T (we use size=750 as about 150 are not unique)
  sample_P_T <- sample(rep(coords$stn_id, (P_T_coords * 1.5e6) ), size = 750, replace = F)
  coords$tst <- (coords$stn_id %in% sample_P_T)*1
  
  # sample 75% from the remaining stations for training
  sample_P_S <- sample(coords[tst != 1, stn_id], nrow(coords[tst != 1,])*0.75)
  coords$trn <- (coords$stn_id %in% sample_P_S)*1
  
  # generating train and test
  coords_trn <- coords[trn==1,]; coords_tst <- coords[tst==1,]
  stn_trn <- coords_trn$stn_id ; stn_tst <- coords_tst$stn_id
  db_trn <- db[stn_id %in% stn_trn,]
  db_tst <- db[stn_id %in% stn_tst,]

  # estimate f_S
  db_pppp <- ppp(x=coords_trn$lon, y=coords_trn$lat, window = ow)
  ker <- density.ppp(db_pppp, edge = T, dimyx = c(grid_n,grid_n), sigma = 0.75)
  all_f_S <- ker$v / sum(ker$v, na.rm = T)
  xindex <- sapply(coords_trn$lon, function(s) which.min(na.omit(abs(ker$xcol-s))))
  yindex <- sapply(coords_trn$lat, function(s) which.min(na.omit(abs(ker$yrow-s))))
  f_S <- numeric(length(coords_trn$lon))
  for (i in seq_along(f_S)) {f_S[i] <- all_f_S[yindex[i],xindex[i]]}
  rownames(all_f_S) <- ker$xcol
  colnames(all_f_S) <- ker$yrow
  
  # assign P_T(g)
  P_T <- numeric(length(coords_trn$lon))
  for (i in seq_along(P_T)) {P_T[i] <- all_P_T[yindex[i],xindex[i]]}
 
  # derive IW
  coords_trn$f_S <- f_S 
  coords_trn$P_T <- P_T
  coords_trn$IW <- coords_trn$P_T/coords_trn$f_S
  coords_trn$IWadj <- ifelse(coords_trn$IW > 1, 1, coords_trn$IW)
  
  # prepare data for learning
  db_trn <- merge(x=db_trn, y=coords_trn[,.(stn_id,IWadj)], by="stn_id")
  shuffled <- sample(1:nrow(db_trn))
  db_trn <- db_trn[shuffled,]
  train_input <- sparse.model.matrix(data = db_trn[,..features ], ~.)
  train_output <- db_trn$tmin
  train_weights <- db_trn$IWadj
  test_input <- sparse.model.matrix(data = db_tst[,..features], ~. )
  test_output <- db_tst$tmin
  
  # models fitting
  # dnn
  m_dnn2 <- build_model()
  history1 <- m_dnn2 %>% fit(
    train_input,
    train_output,
    batch_size = 64,
    epochs = 70,
    validation_data = list(test_input, test_output),
    verbose = verbose )
  m_wdnn2 <- build_model()
  history2 <- m_wdnn2 %>% fit(
    train_input,
    train_output,
    batch_size = 64,
    epochs = 70,
    validation_data = list(test_input, test_output),
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
  m_xgboost2 <- xgb.train(param, dtrain, 2000, watchlist, verbose = verbose)
  m_wxgboost2 <- xgb.train(paramw, dtrain_w, 2000, watchlistw, verbose = verbose)
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
  
  # results
  task2_results[j,] <- c(mse_dnn2, mse_wdnn2, mse_xgboost2, mse_wxgboost2, mse_lmm2, mse_wlmm2)
  
  print(paste("task 2: ", j))
  print(task2_results[j,])
}

write.csv(task2_results, "~/DA/data/task2_results.csv")







##### Task 3 #####

task3_results <- matrix(NA, CV_k, 6)
colnames(task3_results) <- c("mse_dnn", "mse_wdnn", "mse_xgboost", "mse_wxgboost", "mse_lmm", "mse_wlmm")

# load northwest intense P_T and assign P_T(g)
all_P_T <- readRDS( "~/DA/data/task3_all_P_T.Rda")
P_T_coords <- numeric(nrow(coords))
xindex <- sapply(coords$lon, function(s) which.min(abs(as.numeric(colnames(all_P_T))-s)))
yindex <- sapply(coords$lat, function(s) which.min(abs(as.numeric(rownames(all_P_T))-s)))
for (i in seq_along(P_T_coords)) {
  P_T_coords[i] <- all_P_T[yindex[i],xindex[i]]
}

for (j in 1:CV_k) {
  
  # sample approximately 600 stations from P_T (we use size=750 as about 150 are not unique)
  sample_P_T <- sample(rep(coords$stn_id, (P_T_coords * 1.5e6)), size = 750, replace = F)
  coords$tst <- (coords$stn_id %in% sample_P_T)*1
  
  # sample 75% from the remaining stations for training
  sample_P_S <- sample(coords[tst != 1, stn_id], nrow(coords[tst != 1,])*0.75)
  coords$trn <- (coords$stn_id %in% sample_P_S)*1
  
  # generating train and test
  coords_trn <- coords[trn==1,]; coords_tst <- coords[tst==1,]
  stn_trn <- coords_trn$stn_id; stn_tst <- coords_tst$stn_id
  db_trn <- db[stn_id %in% stn_trn,]
  db_tst <- db[stn_id %in% stn_tst,]
  
  # estimate f_S
  db_pppp <- ppp(x=coords_trn$lon, y=coords_trn$lat, window = ow)
  ker <- density.ppp(db_pppp, edge = T, dimyx = c(grid_n,grid_n), sigma = 0.75)
  all_f_S <- ker$v / sum(ker$v, na.rm = T)
  xindex <- sapply(coords_trn$lon, function(s) which.min(na.omit(abs(ker$xcol-s))))
  yindex <- sapply(coords_trn$lat, function(s) which.min(na.omit(abs(ker$yrow-s))))
  f_S <- numeric(length(coords_trn$lon))
  for (i in seq_along(f_S)) {f_S[i] <- all_f_S[yindex[i],xindex[i]]}
  rownames(all_f_S) <- ker$xcol
  colnames(all_f_S) <- ker$yrow
  
  # assign P_T(g)
  P_T <- numeric(length(coords_trn$lon))
  for (i in seq_along(P_T)) {P_T[i] <- all_P_T[yindex[i],xindex[i]]}
  
  # derive IW
  coords_trn$f_S <- f_S
  coords_trn$P_T <- P_T
  coords_trn$IW <- coords_trn$P_T/coords_trn$f_S
  coords_trn$IWadj <- ifelse(coords_trn$IW > 1, 1, coords_trn$IW)
  db_trn <- merge(x=db_trn, y=coords_trn[,.(stn_id,IWadj)], by="stn_id")
  
  # prepare data for learning
  shuffled <- sample(1:nrow(db_trn))
  db_trn <- db_trn[shuffled,]
  train_input <- sparse.model.matrix(data = db_trn[,..features ], ~.) 
  train_output <- db_trn$tmin
  train_weights <- db_trn$IWadj
  test_input <- sparse.model.matrix(data = db_tst[,..features], ~. )
  test_output <- db_tst$tmin

  # models fitting
  # dnn
  m_dnn3 <- build_model()
  history1 <- m_dnn3 %>% fit(
    train_input,
    train_output,
    batch_size = 64,
    epochs = 70, 
    validation_data = list(test_input, test_output),
    verbose = verbose )
  m_wdnn3 <- build_model()
  history3 <- m_wdnn3 %>% fit(
    train_input,
    train_output,
    batch_size = 64,
    epochs = 70, 
    validation_data = list(test_input, test_output),
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
  param <- list(max_depth=6, eta=0.2, verbosity=1)
  paramw <- list(max_depth=6, eta=0.2, verbosity=1, objective=weightedloss, eval_metric=evalerror)
  m_xgboost3 <- xgb.train(param, dtrain, 2000, watchlist, verbose = 0)
  m_wxgboost3 <- xgb.train(paramw, dtrain_w, 2000, watchlistw, verbose = 0)
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

  # results
  task3_results[j,] <- c(mse_dnn3, mse_wdnn3, mse_xgboost3, mse_wxgboost3, mse_lmm3, mse_wlmm3)
  
  print(paste("task 3: ", j))
  print(task3_results[j,])
}

write.csv(task3_results, "~/DA/data/task3_results.csv")


# plot results
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
  geom_point(size = 5, shape = 19) +
  geom_point(data = alls[iw == "no iw",], aes(x=la,y=mean), size = 3, shape = 19, color = "white") +
  theme_bw() + 
  theme(legend.position = "none", strip.text = element_text(size=12), 
        text = element_text(size=12),
        axis.text.x = element_text(size = 12)) + 
  labs(x = "\n Learning Algorithm", y = "Test MSE") 

ggsave("~/DA/charts/results.png")
