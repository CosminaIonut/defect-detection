# from ann_visualizer.visualize import ann_viz
# from keras.optimizers import RMSprop
# from keras.regularizers import l2
#
#
#
# # Create your first MLP in Keras
# from keras.models import Sequential
# from keras.layers import Dense
# import numpy
# # fix random seed for reproducibility
# # numpy.random.seed(7)
# # load pima indians dataset
# # dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# # split into input (X) and output (Y) variables
# # X = dataset[:,0:8]
# # Y = dataset[:,8]
# # create model
#
# model = Sequential()
# model.add(Dense(100,  input_shape=(8,), activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01),
#                   bias_regularizer=l2(0.01)))
# model.add(Dense(20, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01),
#                   bias_regularizer=l2(0.01)))
# model.add(Dense(1, activation='sigmoid'))
# optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# model.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse'])
# # Compile model
# # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # Fit the model
# # model.fit(X, Y, epochs=150, batch_size=10)
# # evaluate the model
# # scores = model.evaluate(X, Y)
# # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#
# ann_viz(model,  filename="baseline3", title="Baseline neural network")
