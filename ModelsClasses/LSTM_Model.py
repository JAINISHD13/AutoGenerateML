import time

import numpy as np
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from numpy import math


class LSTM_Model():

    def __init__(self):
        print("In Class Constructor")

    def lstm_model(self,filepath,arry1D,modelCompileParameters,count):
        global_list = []
        with open('C:/code.txt', 'r') as reader:
            reader.readlines()

        print("Final path:::",filepath)
        file = open('C:/code.txt', "r+")
        file.truncate(0)
        file.close()


        try:
            file = open('c:/filesforproject/model_history_log.csv', "r+")
            file.truncate(0)
            file.close()
        except:
            pass

        with open('C:/code.txt', 'a') as append:
            append.writelines("\nimport numpy as np")
            append.writelines("\nfrom keras.layers.core import Dense, Dropout")
            append.writelines("\nfrom keras.layers.recurrent import LSTM")
            append.writelines("\nfrom sklearn.model_selection import train_test_split")
            append.writelines("\nfrom sklearn.model_selection import train_test_split")
            append.writelines("\nfrom tensorflow.keras.callbacks import EarlyStopping")
            append.writelines("\nfrom keras.callbacks import ModelCheckpoint")
            append.writelines("\nfrom numpy import math")
            append.writelines("\n\nclass LSTM_Model():")
            append.writelines("\n\tdef __init__(self):")
            append.writelines("\n\t\tprint(""In Class Constructor"")")
            append.writelines("\n\t\tdef lstm_model(self,filepath,arry1D,modelCompileParameters,count):")
            append.writelines("\n\t\t\tdataset = np.loadtxt(filepath)")
            append.writelines("\n\t\t\tload_original_arr = dataset.reshape(dataset.shape[0], dataset.shape[1] // int(column[0]), int(column[1]))")
            append.writelines("\n\t\t\tx_val = load_original_arr")
            append.writelines("\n\t\t\ty_val = np.loadtxt(""path/ydata.csv"")")
            append.writelines("\n\t\t\tx_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size="+modelCompileParameters[10]+")")
            append.writelines("\n\t\t\tmodel = Sequential()")
            append.writelines("")
            append.writelines("")
            append.writelines("")


        column = arry1D[0][3].split(",")
        dataset = np.loadtxt(filepath)
        load_original_arr = dataset.reshape(dataset.shape[0], dataset.shape[1] // int(column[0]), int(column[1]))

        # dataset = np.loadtxt("C:/Users/Jainish A Dabhi/Desktop/Data Visualization/xstockdata.csv")
        # load_original_arr = dataset.reshape(dataset.shape[0], dataset.shape[1] // 5, 5)

        x_val = load_original_arr
        print(x_val.shape)

        y_val = np.loadtxt("C:/Users/Jainish A Dabhi/Desktop/Data Visualization/ystockdata.csv")

        x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size=float(modelCompileParameters[10]))

        # x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size=0.30)

        model = Sequential()

        i=0
        with open('C:/code.txt', 'a') as append:

            while(i<count):

                if i==0:
                    splitShape = arry1D[0][1].split(',')
                    append.writelines("\n\t\t\tmodel.add(LSTM("+str(arry1D[i][0])+", input_shape=("+str(splitShape[0])+","+ str(splitShape[1])+"), return_sequences="+str(bool(arry1D[i][5]))+")")
                    append.writelines("\n\t\t\t model.add(Dropout("+arry1D[i][2]+"))")
                    model.add(LSTM(int(arry1D[i][0]), input_shape=(int(splitShape[0]), int(splitShape[1])), return_sequences=arry1D[i][5]))
                    model.add(Dropout(float(arry1D[i][2])))
                else:
                    splitShape = arry1D[i][1].split(',')
                    print("splitshape:",splitShape)
                    print("splitshape:", type(splitShape))
                    # model.add(LSTM(128, input_shape=(7, 5),return_sequences=True))
                    append.writelines("\n\t\t\t model.add(LSTM( "+arry1D[i][0]+" ,input_shape=( "+splitShape[0]+","+splitShape[1]+" ), return_sequences=True )")
                    model.add(LSTM(int(arry1D[i][0]), input_shape=(int(splitShape[0]), int(splitShape[1])),return_sequences=True))
                    append.writelines("\n\t\t\tmodel.add(Dropout(" +arry1D[i][2]+ "))")
                    model.add(Dropout(float(arry1D[i][2])))
                i=i+1


        # model.add(LSTM(128, input_shape=(7, 5), return_sequences=True))
        # model.add(Dropout(0.2))
        # model.add(LSTM(128, input_shape=(7, 5)))
        # model.add(Dropout(0.2))
        i=0
        with open('C:/code.txt', 'a') as append:

            while(i<count):
                if i==0:
                    append.writelines("\n\t\t\tmodel.add(Dense("+arry1D[0][4]+"))")
                    model.add(Dense(int(arry1D[0][4])))
                else:
                    append.writelines("\n\t\t\tmodel.add(Dense("+arry1D[i][3]+"))")
                    model.add(Dense(int(arry1D[i][3])))
                i=i+1

        # adam = keras.optimizers.Adam(decay=0.2)

        with open('C:/code.txt', 'a') as append:
            append.writelines("\n\t\t\tmodel.compile(loss="+modelCompileParameters[5]+",optimizer="+modelCompileParameters[7]+", metrics=["+modelCompileParameters[6]+"])")
            append.writelines("\n\t\t\tcheckpoint = ModelCheckpoint(filepath='best_model_Lstm_1.h5', monitor='val_loss', save_best_only=True)")
            append.writelines("\n\t\t\tmonitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=4, verbose=1, mode='auto')")
            append.writelines("\n\t\t\tprint('Train...')")
            append.writelines("\n\t\t\tfrom keras.callbacks import CSVLogger")
            append.writelines("\n\t\t\tcsv_logger = CSVLogger(""filepath/model_history_log.csv"", append=True)")
            append.writelines("\n\t\t\tmodel.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[csv_logger,checkpoint, monitor], epochs="+modelCompileParameters[1]+",verbose="+modelCompileParameters[2]+")")
            append.writelines("\n\t\t\t\tdef model_score(model, x_train, y_train, x_test, y_test):")
            append.writelines("\n\t\t\t\t\ttrainScore = model.evaluate(x_train, y_train, verbose=0)")
            append.writelines("\n\t\t\t\t\tprint('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))")
            append.writelines("\n\t\t\t\t\ttestScore = model.evaluate(x_test, y_test, verbose=0)")
            append.writelines("\n\t\t\t\t\tprint('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))")
            append.writelines("\n\t\t\t\t\treturn trainScore[0], testScore[0]")
            append.writelines("\n\t\t\tmodel_score(model, x_train, y_train, x_test, y_test)")
            append.writelines("\n\t\t\timport matplotlib.pyplot as plt2")
            append.writelines("\n\t\t\tp = model.predict(x_test)")
            append.writelines("\n\t\t\tnp1 = np.sort(p, axis=None)")
            append.writelines("\n\t\t\tny_test = np.sort(y_test, axis=None)")
            append.writelines("\n\t\t\tplt2.plot(np1, color='red', label='Prediction')")
            append.writelines("\n\t\t\tplt2.plot(ny_test, color='blue', label='Actual')")
            append.writelines("\n\t\t\tplt2.legend(loc='best')")
            append.writelines("\n\t\t\tplt2.gcf()")
            append.writelines("\n\t\t\tplt2.show()")
            append.writelines("")
            append.writelines("")


        model.compile(loss=modelCompileParameters[5], optimizer=modelCompileParameters[7], metrics=[modelCompileParameters[6]])
        # print("Compilation Time : ", time.time() - start)

        from keras.callbacks import ModelCheckpoint

        checkpoint = ModelCheckpoint(filepath='best_model_Lstm_1.h5', monitor='val_loss', save_best_only=True)
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=4, verbose=1, mode='auto')
        print('Train...')
        # ,callbacks=[monitor]


        from keras.callbacks import CSVLogger
        csv_logger = CSVLogger("c:/filesforproject/model_history_log.csv", append=True)

        t1 = time.time()

        model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[csv_logger, checkpoint, monitor],
                  epochs=int(modelCompileParameters[1]),
                  verbose=int(modelCompileParameters[2]))

        train_time = time.time() - t1

        def model_score(model, x_train, y_train, x_test, y_test):
            trainScore = model.evaluate(x_train, y_train, verbose=0)
            print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

            testScore = model.evaluate(x_test, y_test, verbose=0)
            print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
            global_list.append(math.sqrt(trainScore[0]))
            global_list.append(math.sqrt(testScore[0]))
            print("global_list_method:",global_list)
            return trainScore[0], testScore[0]



        model_score(model, x_train, y_train, x_test, y_test)

        import matplotlib.pyplot as plt2

        p = model.predict(x_test)

        np1 = np.sort(p, axis=None)
        ny_test = np.sort(y_test, axis=None)

        plt2.plot(np1, color='red', label='Prediction')
        plt2.plot(ny_test, color='blue', label='Actual')
        plt2.legend(loc='best')
        plt2.savefig("c:/graphs/fig.png", dpi=100)
        plt2.gcf()
        plt2.show()
        plt2.savefig("c:/graphs/fig.pdf", bbox_inches="tight", pad_inches=2, transparent=True)

        print("global_list:",global_list)
        return global_list