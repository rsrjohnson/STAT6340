library(keras)


#Experiment 1
print("Experiment 1")

layer_sizes=list(NN1=512,NN2=512,NN3=256,NN4=256,NN5=c(512,512),NN6=c(512,512),
                 NN7=c(256,256),NN8=c(256,256),NN9=512,NN10=512)

error.df=data.frame(Train_Error=rep(0,10),Test_Error=rep(0,10),HiddenLayers=matrix(layer_sizes))


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


hl_sizes=c(512,512,256,256)
epc=c(5,10,5,10)

#Questions 1a-1d
for(i in seq(1,4))
{
  nn <- keras_model_sequential() %>%
    layer_dense(units = hl_sizes[i], activation = "relu", input_shape = c(28*28)) %>%
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


#Questions 1e-1h
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


#Questions 1i

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

nn.reg %>% fit(train_images, train_labels, epochs = 10, batch_size = 128)

metrics_trn=nn.reg %>% evaluate(train_images, train_labels)

metrics_tst=nn.reg %>% evaluate(test_images, test_labels)

error.df[9,"Train_Error"]=1-metrics_trn[[2]]
error.df[9,"Test_Error"]=1-metrics_tst[[2]]


#Questions 1j

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

### Preprocess the data

### Standardize the training and test features

mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)

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