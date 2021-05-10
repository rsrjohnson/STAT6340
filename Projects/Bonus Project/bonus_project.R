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


