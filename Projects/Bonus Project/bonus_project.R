#Seed to replicate results
rdseed=8466
set.seed(rdseed)

library(ggplot2)
library(keras)
library(tensorflow)

#Setting the seed for tensorflow session
tf$random$set_seed(rdseed)



#Experiment 1
print("Experiment 1")

#Number of Epochs for 1a-1d, 1e-1h
epc=c(5,10,5,10)

layer_sizes=list(NN1=512,NN2=512,NN3=256,NN4=256,NN5=c(512,512),NN6=c(512,512),
                 NN7=c(256,256),NN8=c(256,256),NN9=512,NN10=512)


error.df=data.frame(Train_Error=rep(0,10),Test_Error=rep(0,10),
                    HiddenLayers=matrix(layer_sizes),
                    Epochs=c(epc,epc,5,5),
                    Regularization=c(rep("NO",8),"L2","NO"),
                    DropOut=c(rep("NO",9),"50%"))


#Importing data set
mnist = dataset_mnist()
train_images = mnist$train$x
train_labels = mnist$train$y
test_images = mnist$test$x
test_labels = mnist$test$y

train_images = array_reshape(train_images, c(60000, 28*28)) # matrix
train_images = train_images/255 # ensures all values are in [0, 1]
test_images = array_reshape(test_images, c(10000, 28*28))
test_images = test_images/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)




####Questions 1a-1d####

#Hidden Layers sizes for 1a-1d
hl_sizes=c(512,512,256,256)

for(i in seq(1,4))
{
  nn <- keras_model_sequential() %>%
    layer_dense(units = hl_sizes[i], activation = "relu", input_shape = c(28*28) ) %>%
    layer_dense(units = 10, activation = "softmax")
  
  nn %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",  # loss function to minimize
    metrics = c("accuracy") # monitor classification accuracy
  )
  
  
  
  nn %>% fit(train_images, train_labels, epochs = epc[i], batch_size = 128)
  
  metrics_trn=nn %>% evaluate(train_images, train_labels)
  
  metrics_tst=nn %>% evaluate(test_images, test_labels)
  
  error.df[i,"Train_Error"]=1-metrics_trn[[2]]
  error.df[i,"Test_Error"]=1-metrics_tst[[2]]
  
}


####Questions 1e-1h####

#Hidden Layers sizes for 1e-1h
hl1=c(512,512,256,256)
hl2=c(512,512,256,256)

for(i in seq(1,4))
{
  nn <- keras_model_sequential() %>%
    layer_dense(units = hl1[i], activation = "relu", input_shape = c(28*28)) %>%
    layer_dense(units = hl2[i], activation = "relu") %>%
    layer_dense(units = 10, activation = "softmax")
  
  nn %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",  # loss function to minimize
    metrics = c("accuracy") # monitor classification accuracy
  )
  
  
  
  nn %>% fit(train_images, train_labels, epochs = epc[i], batch_size = 128)
  
  metrics_trn=nn %>% evaluate(train_images, train_labels)
  
  metrics_tst=nn %>% evaluate(test_images, test_labels)
  
  error.df[i+4,"Train_Error"]=1-metrics_trn[[2]]
  error.df[i+4,"Test_Error"]=1-metrics_tst[[2]]
  
}


####Questions 1i####

### Regularize the weights using ridge penalty with specified lambda

nn.reg=keras_model_sequential() %>% 
  layer_dense(units = 512, activation = "relu", input_shape = c(28*28),
              kernel_regularizer = regularizer_l2(0.001)) %>%
   layer_dense(units = 10, activation = "softmax")

### Compile

nn.reg %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

nn.reg %>% fit(train_images, train_labels, epochs = 5, batch_size = 128)

metrics_trn=nn.reg %>% evaluate(train_images, train_labels)

metrics_tst=nn.reg %>% evaluate(test_images, test_labels)

error.df[9,"Train_Error"]=1-metrics_trn[[2]]
error.df[9,"Test_Error"]=1-metrics_tst[[2]]


####Questions 1j####

nn.drop <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = c(28*28)) %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 10, activation = "softmax")

nn.drop %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",  # loss function to minimize
  metrics = c("accuracy") # monitor classification accuracy
)

nn.drop %>% fit(train_images, train_labels, epochs = 5, batch_size = 128)

metrics_trn=nn %>% evaluate(train_images, train_labels)

metrics_tst=nn %>% evaluate(test_images, test_labels)

error.df[10,"Train_Error"]=1-metrics_trn[[2]]
error.df[10,"Test_Error"]=1-metrics_tst[[2]]





#Experiment 2
print("Experiment 2")

boston <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% boston

layer_sizes2=list(NN1=c(64,64),NN2=128,NN3=c(64,64),NN4=128)

error.df2=data.frame(Val_MAE=rep(0,4),Test_MAE=rep(0,4),
                     HiddenLayers=matrix(layer_sizes2),
                     Regularization=c("NO","NO","L2","L2"))

### Preprocess the data

### Standardize the training and test features

mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)





####Questions 2a####


build_model <- function(){
  # specify the model
  model <- keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu", 
                input_shape = dim(train_data[[2]])) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)
  # compile the model
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae") # mean absolute error
  )
}


# K-fold CV

k <- 4
set.seed(8466)
indices <- sample(1:nrow(train_data))
folds <- cut(indices, breaks = k, labels = FALSE)


num_epochs <- 200
all_mae_histories <- NULL
for (i in 1:k){
  cat("Processing fold #", i, "\n")
  
  val_indices <- which(folds == i, arr.ind = TRUE) # Prepares the validation data, data from partition #k
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  model <- build_model()
  
  history <- model %>% fit(
    partial_train_data, partial_train_targets,
    validation_data = list(val_data, val_targets),
    epochs = num_epochs, batch_size = 64, verbose = 0
  )
  
  mae_history <- history$metrics$val_mae
  all_mae_histories <- rbind(all_mae_histories, mae_history)
}

### Get the mean K-fold validation MAE for each epoch
average_mae_history <- data.frame(
  epoch = seq(1:ncol(all_mae_histories)),
  validation_mae = apply(all_mae_histories, 2, mean)
)

best_epoch=which.min(average_mae_history$validation_mae)

### Plot the history
print(ggplot(average_mae_history,aes(x=epoch,y=validation_mae))+geom_line()+
  ylim(2,5)+geom_point(aes(x=best_epoch,y=validation_mae[best_epoch]),
                       color="red",size=3,shape=20))

final_e=75

maer=rep(0,k)

for (i in 1:k){
  cat("Processing fold #", i, "\n")
  
  val_indices <- which(folds == i, arr.ind = TRUE) # Prepares the validation data, data from partition #k
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  model <- build_model()
  
  res_i=model %>% fit(
    partial_train_data, partial_train_targets,
    validation_data = list(val_data, val_targets),
    epochs = final_e, batch_size = 16, verbose = 0
  )
  
  res_i$metrics
  metrics_trn=model %>% evaluate(val_data, val_targets)
  
  maer[i]=metrics_trn[[2]]
  
}


error.df2[1,"Val_MAE"]=mean(maer)

#Fitting with all training data
model <- build_model()
model %>% fit(train_data, train_targets, epochs = final_e, 
              batch_size = 16, verbose = 0)

metrics_tst=model %>% evaluate(test_data, test_targets)

error.df2[1,"Test_MAE"]=metrics_tst[[2]]



####Questions 2b####



build_model2 <- function(){
  # specify the model
  model <- keras_model_sequential() %>% 
    layer_dense(units = 128, activation = "relu", 
                input_shape = dim(train_data[[2]])) %>%
    layer_dense(units = 1)
  # compile the model
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae") # mean absolute error
  )
}


maer=rep(0,k)

for (i in 1:k){
  cat("Processing fold #", i, "\n")
  
  val_indices <- which(folds == i, arr.ind = TRUE) # Prepares the validation data, data from partition #k
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  model <- build_model2()
  
  res_i=model %>% fit(
    partial_train_data, partial_train_targets,
    validation_data = list(val_data, val_targets),
    epochs = final_e, batch_size = 16, verbose = 0
  )
  
  res_i$metrics
  metrics_trn=model %>% evaluate(val_data, val_targets)
  
  maer[i]=metrics_trn[[2]]
  
}


error.df2[2,"Val_MAE"]=mean(maer)

#Fitting with all training data
model <- build_model2()
model %>% fit(train_data, train_targets, epochs = final_e, 
              batch_size = 16, verbose = 0)

metrics_tst=model %>% evaluate(test_data, test_targets)

error.df2[2,"Test_MAE"]=metrics_tst[[2]]


####Questions 2c####

build_model3 <- function(){
  # specify the model
  model <- keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu", 
                input_shape = dim(train_data[[2]]),
                kernel_regularizer = regularizer_l2(0.001)) %>%
    layer_dense(units = 64, activation = "relu",
                kernel_regularizer = regularizer_l2(0.001)) %>%
    layer_dense(units = 1)
  # compile the model
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae") # mean absolute error
  )
}


maer=rep(0,k)

for (i in 1:k){
  cat("Processing fold #", i, "\n")
  
  val_indices <- which(folds == i, arr.ind = TRUE) # Prepares the validation data, data from partition #k
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  model <- build_model3()
  
  res_i=model %>% fit(
    partial_train_data, partial_train_targets,
    validation_data = list(val_data, val_targets),
    epochs = final_e, batch_size = 16, verbose = 0
  )
  
  res_i$metrics
  metrics_trn=model %>% evaluate(val_data, val_targets)
  
  maer[i]=metrics_trn[[2]]
  
}


error.df2[3,"Val_MAE"]=mean(maer)

#Fitting with all training data
model <- build_model3()
model %>% fit(train_data, train_targets, epochs = final_e, 
              batch_size = 16, verbose = 0)

metrics_tst=model %>% evaluate(test_data, test_targets)

error.df2[3,"Test_MAE"]=metrics_tst[[2]]


####Questions 2d####


build_model4 <- function(){
  # specify the model
  model <- keras_model_sequential() %>% 
    layer_dense(units = 128, activation = "relu", 
                input_shape = dim(train_data[[2]]),
                kernel_regularizer = regularizer_l2(0.001)) %>%
    layer_dense(units = 1)
  # compile the model
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae") # mean absolute error
  )
}

maer=rep(0,k)

for (i in 1:k){
  cat("Processing fold #", i, "\n")
  
  val_indices <- which(folds == i, arr.ind = TRUE) # Prepares the validation data, data from partition #k
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  model <- build_model3()
  
  res_i=model %>% fit(
    partial_train_data, partial_train_targets,
    validation_data = list(val_data, val_targets),
    epochs = final_e, batch_size = 16, verbose = 0
  )
  
  res_i$metrics
  metrics_trn=model %>% evaluate(val_data, val_targets)
  
  maer[i]=metrics_trn[[2]]
  
}


error.df2[4,"Val_MAE"]=mean(maer)

#Fitting with all training data
model <- build_model4()
model %>% fit(train_data, train_targets, epochs = final_e, 
              batch_size = 16, verbose = 0)

metrics_tst=model %>% evaluate(test_data, test_targets)

error.df2[4,"Test_MAE"]=metrics_tst[[2]]