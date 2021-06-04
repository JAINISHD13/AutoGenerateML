from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
import seaborn as sn

plt.style.use('ggplot')

class ConvolutionNN():

    def __init__(self):
        print("In Class Constructor")


    def convolutionNN(self,filepath,arry1D,modelCompileParameters,count):
        print("In CNN CLASS")
        global_list = []

        with open('C:/code.txt', 'r') as reader:
            reader.readlines()

        print("Final path:::",filepath)
        file = open('C:/code.txt', "r+")
        file.truncate(0)
        file.close()

        file = open('c:/filesforproject/model_history_log.csv', "r+")
        file.truncate(0)
        file.close()

        with open('C:/code.txt', 'a') as append:
            append.writelines("\nfrom keras.models import Sequential")
            append.writelines("\nfrom keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense")
            append.writelines("\nfrom keras.utils.np_utils import to_categorical")
            append.writelines("\nimport matplotlib.pyplot as plt")
            append.writelines("\nimport pandas as pd")
            append.writelines("\nimport numpy as np")
            append.writelines("\nfrom sklearn.model_selection import train_test_split")
            append.writelines("\nplt.style.use('ggplot')")
            append.writelines("\n\nclass ConvolutionNN():")
            append.writelines("\n\tdef __init__(self):")
            append.writelines("\n\tdef convolutionNN(self,filepath,arry1D,modelCompileParameters,count):")
            append.writelines("\n\t\ttrainData = pd.read_csv(filepath)")
            append.writelines("\n\t\ttestData = pd.read_csv(filepath)")
            append.writelines("\n\t\tY_train = trainData[""label""]")
            append.writelines("\n\t\tX_train = trainData.drop(labels=[""label""], axis=1)")
            append.writelines("\n\t\tY_test = testData[""label""]")
            append.writelines("\n\t\tX_test = testData.drop(labels=[""label""], axis=1)")
            append.writelines("\n\t\tX_train = X_train / 255.0")
            append.writelines("\n\t\tX_test = X_test / 255.0")


        trainData = pd.read_csv(filepath)
        testData = pd.read_csv(filepath)


        trainData.head()

        print("Train Data Shape: ", trainData.shape)
        print("Test Data Shape: ", testData.shape)

        Y_train = trainData[modelCompileParameters[8]]
        X_train = trainData.drop(labels=[modelCompileParameters[8]], axis=1)


        # Y_train = trainData["label"]
        # X_train = trainData.drop(labels=["label"], axis=1)

        print(X_train.shape)

        # Y_test = testData["label"]
        # X_test = testData.drop(labels=["label"], axis=1)

        X_train = X_train / 255.0
        # X_test = X_test / 255.0
        reshape_para = []
        reshape_para = modelCompileParameters[0].split(',')

        with open('C:/code.txt', 'a') as append:
            append.writelines("\n\t\tX_train = X_train.values.reshape(-1, 28, 28, 1)")
            append.writelines("\n\t\tX_test = X_test.values.reshape("+str(reshape_para[0])+","+str(reshape_para[1])+","+str(reshape_para[2])+","+str(reshape_para[3])+")")
            append.writelines("\n\t\tY_train = to_categorical(Y_train, num_classes="+modelCompileParameters[9]+")")
            append.writelines("\n\t\tx_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.33, random_state=0)")
            append.writelines("\n\t\tmodel = Sequential()")
            append.writelines("")


        X_train = X_train.values.reshape(int(reshape_para[0]), int(reshape_para[1]), int(reshape_para[2]), int(reshape_para[3]))
        # X_test = X_test.values.reshape(int(reshape_para[0]), int(reshape_para[1]), int(reshape_para[2]), int(reshape_para[3]))

        # X_train = X_train.values.reshape(-1, 28, 28, 1)
        # X_test = X_test.values.reshape(-1, 28, 28, 1)

        print("X Train Shape: ", X_train.shape)
        # print("X Test Shape: ", X_test.shape)

        img = np.asmatrix(X_train[4])
        img = img.reshape((int(reshape_para[1]), int(reshape_para[2])))  # 28*28=784

        # img = np.asmatrix(X_train[4])
        # img = img.reshape((28, 28))  # 28*28=784
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.show()

        Y_train = to_categorical(Y_train, num_classes=int(modelCompileParameters[9]))

        # Y_train = to_categorical(Y_train, num_classes=10)
        print(Y_train.shape)

        x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=float(modelCompileParameters[10]), random_state=0)

        # x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.33, random_state=0)

        model = Sequential()

        #0->padding
        #1->activation
        #2->filter
        #3->dropout
        #4->kernel
        #5->dense
        #6->inputshape
        #7->dropoutafter

        i = 0
        with open('C:/code.txt', 'a') as append:
            while (i < count):
                ker = []
                ker = arry1D[i][4].split(',')
                if (i == 0):
                    shape = []
                    shape = arry1D[0][5].split(',')
                    append.writelines("\n\t\tmodel.add(Conv2D(filters="+arry1D[0][2]+",kernel_size="+ker[1]+",padding="+arry1D[0][0]+",input_shape=("+shape[0]+","+shape[1]+","+shape[2]+"))")
                    append.writelines("\n\t\tActivation("+arry1D[0][1]+")")
                    append.writelines("\n\t\tmodel.add(MaxPooling2D())")
                    append.writelines("\n\t\tmodel.add("+arry1D[0][3]+")")

                    model.add(Conv2D(filters=int(arry1D[0][2]),kernel_size=(int(ker[0]),int(ker[1])), padding=arry1D[0][0],input_shape=(int(shape[0]),int(shape[1]),int(shape[2]))))
                    model.add(Activation(arry1D[0][1]))
                    model.add(MaxPooling2D())
                    model.add(Dropout(float(arry1D[0][3])))
                else:
                    append.writelines("\n\t\tmodel.add(Conv2D(filters=" + arry1D[i][2] + "," + ker[1] + ",padding=" + arry1D[i][0])
                    append.writelines("\n\t\tActivation(" + arry1D[i][1] + ")")
                    append.writelines("\n\t\tmodel.add(MaxPooling2D())")
                    append.writelines("\n\t\tmodel.add(" + arry1D[i][3] + ")")

                    model.add(Conv2D(int(arry1D[i][2]),(int(ker[0]),int(ker[1])), padding=arry1D[i][0]))
                    model.add(Activation(arry1D[i][1]))
                    model.add(MaxPooling2D())
                    model.add(Dropout(float(arry1D[0][3])))
                i = i + 1;


        # model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", input_shape=(28, 28, 1)))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D())
        # model.add(Dropout(0.25))
        #
        # model.add(Conv2D(32, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D())
        # model.add(Dropout(0.25))
        #
        # model.add(Conv2D(64, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D())
        # model.add(Dropout(0.25))

        with open('C:/code.txt', 'a') as append:
            append.writelines("\n\t\tmodel.add(Flatten())")
        model.add(Flatten())

        j=0
        with open('C:/code.txt', 'a') as append:
            while(j<count):
                if j==0:
                    append.writelines("\n\t\tDense(units="+arry1D[0][6]+")")
                    append.writelines("\n\t\tDropout("+arry1D[0][7]+")")
                    append.writelines("\n\t\tActivation("+arry1D[0][8]+")")
                    model.add(Dense(units=int(arry1D[0][6])))
                    model.add(Dropout(float(arry1D[0][7])))
                    model.add(Activation(arry1D[0][8]))
                elif j==count:
                    append.writelines("\n\n\t\tDense(units="+arry1D[j][5]+")")
                    append.writelines("\n\t\tActivation("+arry1D[j][7]+")")
                    model.add(Dense(int(arry1D[j][5])))
                    model.add(Activation(arry1D[j][7]))
                else:
                    append.writelines("\n\n\t\tDense(units="+arry1D[j][5]+")")
                    append.writelines("\n\t\tDropout("+arry1D[j][6]+")")
                    append.writelines("\n\t\tActivation("+arry1D[j][7]+")")
                    model.add(Dense(int(arry1D[j][5])))
                    model.add(Dropout(float(arry1D[j][6])))
                    model.add(Activation(arry1D[j][7]))
                j = j+1



        # model.add(Dense(units=512))
        # model.add(Activation("relu"))
        # model.add(Dropout(0.5))
        #
        # model.add(Dense(512))
        # model.add(Activation("relu"))
        # model.add(Dropout(0.5))
        #
        # model.add(Dense(10))
        # model.add(Activation("softmax"))

        epochs_final  = int(modelCompileParameters[1])
        batchsize = int(modelCompileParameters[2])
        loss_function = modelCompileParameters[5]
        optimizer_function = modelCompileParameters[7]
        metrics_function = modelCompileParameters[6]

        with open('C:/code.txt', 'a') as append:
            append.writelines("\n\t\tmodel.compile(loss="+loss_function+",optimizer="+optimizer_function+",metrics=["+metrics_function+"])")
            model.compile(loss = loss_function,optimizer=optimizer_function,metrics=[metrics_function])
        # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


        # datagenTrain = ImageDataGenerator(
        #     shear_range=0.3,
        #     horizontal_flip=True,
        #     zoom_range=0.3)
        from keras.callbacks import CSVLogger
        csv_logger = CSVLogger("c:/filesforproject/model_history_log.csv", append=True)

        with open('C:/code.txt', 'a') as append:
            append.writelines("\n\t\thistory = model.fit(x_train,y_train, batch_size="+str(batchsize)+",epochs="+str(epochs_final)+",validation_data=(x_val, y_val)")
            append.writelines("\n\t\tloss_train = history.history['loss']")
            append.writelines("\n\t\tloss_val = history.history['val_loss']")
            append.writelines("\n\t\tepochs = range(1, "+str((epochs_final + 1))+")")
            append.writelines("\n\t\tplt.plot(epochs, loss_train, 'g', label='Training loss')")
            append.writelines("\n\t\tplt.plot(epochs, loss_val, 'b', label='validation loss')")
            append.writelines("\n\t\tplt.title('Training and Validation loss')")
            append.writelines("\n\t\tplt.xlabel('Epochs')")
            append.writelines("\n\t\tplt.ylabel('Loss')")
            append.writelines("\n\t\tplt.legend() \n\n\t\t plt.gcf() \n\n\t\tplt.show() ")

        history = model.fit(x_train,y_train , batch_size=batchsize, epochs=epochs_final,
                            validation_data=(x_val, y_val),callbacks = [csv_logger])
        print("Epochs_finalL:",epochs_final)
        # datagenTrain.fit(x_train)
        # history1 = model.fit_generator(datagenTrain.flow(x_train,y_train,batch_size=batchsize),
        #                               epochs=epochs_final, validation_data=(x_val,y_val),
        #                               steps_per_epoch=x_train.shape[0]//batchsize)

        with open('C:/code.txt', 'a') as append:
            append.writelines("\n\t\tplt.plot(history.history[""loss""], label=""Train Loss"")")
            append.writelines("\n\t\tplt.plot(history.history[""val_loss""], label=""Validation Loss"")")
            append.writelines("\n\t\tplt.legend()")
            append.writelines("\n\t\tplt.figure()")
            append.writelines("\n\t\tplt.plot(history.history[""accuracy""], label=""Train Accuracy"")")
            append.writelines("\n\t\tplt.plot(history.history[""val_accuracy""], label=""Validation Accuracy"")")
            append.writelines("\n\t\tplt.legend()")

        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.legend()

        plt.figure()
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.legend()

        loss_train = history.history['loss']
        loss_val = history.history['val_loss']

        epochs = range(1, epochs_final + 1)  # epoch+1

        plt.plot(epochs, loss_train, 'g', label='Training loss')
        plt.plot(epochs, loss_val, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("c:/graphs/fig1.png", dpi=100)
        plt.gcf()
        plt.show()
        plt.savefig("c:/graphs/fig1.pdf", bbox_inches="tight", pad_inches=2, transparent=True)

        with open('C:/code.txt', 'a') as append:
            append.writelines("\n\t\tresults = model.evaluate(x_val,y_val)")
            append.writelines("\n\t\ty_pred = model.predict(x_val)")
            append.writelines("\n\t\trows = len(y_pred)")
            append.writelines("\n\t\tcols= len(y_pred[0])")
            append.writelines("\n\t\tfor i in range(rows):")
            append.writelines("\n\t\t\tfor j in range(cols):")
            append.writelines("\n\t\t\t\tif (y_pred[i][j] > 0.98):")
            append.writelines("\n\t\t\t\t\ty_pred[i][j] = 1")
            append.writelines("\n\t\t\t\telse:")
            append.writelines("\n\t\t\t\t\ty_pred[i][j] = 0")
            append.writelines("\n\t\tprint(""Classification Report:\n%s\n""%(metrics.classification_report(y_val,y_pred)))")
            append.writelines("\n\t\tconf_metrics = metrics.confusion_matrix(y_val.argmax(axis=1),y_pred.argmax(axis=1))")
            append.writelines("\n\t\tdf_cm = pd.DataFrame(conf_metrics,range(int(modelCompileParameters[9])),range(int(modelCompileParameters[9])))")
            append.writelines("\n\t\tplt.figure(figsize=(10,7))")
            append.writelines("\n\t\tsn.set(font_scale=1.4)")
            append.writelines("\n\t\tsn.heatmap(df_cm,fmt="".0f"",annot=True,annot_kws={""size"":16})")
            append.writelines("\n\t\tplt.gcf()")
            append.writelines("\n\t\tplt.show()")


        results = model.evaluate(x_val,y_val)

        print("Accuracy:",results[1])

        y_pred = model.predict(x_val)
        rows = len(y_pred)
        cols= len(y_pred[0])

        for i in range(rows):
            for j in range(cols):
                if (y_pred[i][j] > 0.98):
                    y_pred[i][j] = 1
                else:
                    y_pred[i][j] = 0

        print("metrics:",metrics.classification_report(y_val,y_pred))
        print("Classification Report:\n%s\n"%
                           (metrics.classification_report(y_val,y_pred)))

        precision, recall, fscore, support = score(y_val, y_pred, average='macro')

        # report = metrics.classification_report(y_val, y_pred,output_dict=True)
        # df = pd.DataFrame(report).transpose()
        # df.to_csv('c:/filesforproject/classification_report.csv', index=True)
        #
        conf_metrics = metrics.confusion_matrix(y_val.argmax(axis=1),y_pred.argmax(axis=1))
        print(conf_metrics)
        df_cm = pd.DataFrame(conf_metrics,range(int(modelCompileParameters[9])),range(int(modelCompileParameters[9])))
        plt.figure(figsize=(10,7))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm,fmt=".0f", cmap="Blues",annot=True,annot_kws={"size":16})
        plt.savefig("c:/graphs/fig.png", dpi=100)
        plt.gcf()
        plt.show()
        plt.savefig("c:/graphs/fig.pdf", bbox_inches="tight", pad_inches=2, transparent=True)

        # f1 = f1_score(y_val, y_pred, average=None)
        loss = results[0]
        acc = results[1]
        global_list.append(fscore)
        global_list.append(acc)
        global_list.append(loss)
        global_list.append(precision)
        global_list.append(recall)
        global_list.append(support)

        return global_list

        # print('F1-Score - ', f1_score(y_val, y_pred, average=None))
        # batch_size = 32

        # history = model.fit_generator(datagenTrain.flow(x_train, y_train, batch_size=batchsize),
        #                               epochs=10, validation_data=(x_val, y_val),
        #                               steps_per_epoch=x_train.shape[0] // batch_size)

