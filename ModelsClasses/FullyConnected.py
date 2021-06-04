import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import warnings
import pandas as pd
from numpy import  array
plt.style.use('ggplot')


class FullyConnected():

    def __init__(self):
        print("In Class Constructor")

    def fullyConnectedModel(self,filepath,arry1D,modelCompileParameters,count):
        global_list = []
        print("In fully Connected Model Method")
        with open('C:/code.txt', 'r') as reader:
            reader.readlines()

        print("Final path:::",filepath)
        file = open('C:/code.txt', "r+")
        file.truncate(0)
        file.close()

        file = open('c:/filesforproject/model_history_log.csv', "r+")
        file.truncate(0)
        file.close()

        # with open('C:/code.txt', 'a') as append:
        #     append.writelines("\n Code is reviewed")

        # iris_data = filepath


        data_set = pd.read_csv(filepath)
        print(data_set.head(10))
        # iris_data = load_iris()

        with open('C:/code.txt', 'a') as append:
            append.writelines("import matplotlib.pyplot as plt")
            append.writelines("\nfrom keras.layers import Dense")
            append.writelines("\nfrom keras.models import Sequential")
            append.writelines("\nfrom keras.optimizers import Adam")
            append.writelines("\nfrom sklearn.datasets import load_iris")
            append.writelines("\nfrom sklearn.metrics import f1_score, classification_report")
            append.writelines("\nfrom sklearn.model_selection import train_test_split")
            append.writelines("\nfrom sklearn.preprocessing import OneHotEncoder")
            append.writelines("\nplt.style.use('ggplot')")
            append.writelines("\nclass FullyConnected():")
            append.writelines("\ntdef __init__(self):\n")
            append.writelines("\n\tdef fullyConnectedModel(self,arry1D,modelCompileParameters,count):")
            append.writelines("\n\t\tiris_data = load_iris()")
            append.writelines("\n\t\tx = iris_data.data")
            append.writelines("\n\t\tiris_data.target.reshape(-1, 1)")
            append.writelines("\n\t\tencoder = OneHotEncoder(sparse=False)")
            append.writelines("\n\t\ty = encoder.fit_transform(y_)")
            append.writelines("\n\t\ttrain_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)")
            append.writelines("\n\t\tmodel = Sequential()")

        print('Example data: ')
        # print(iris_data.data[:5])
        # print('Example labels: ')
        # print(iris_data.target[:5])

        inputShape = int(modelCompileParameters[0])
        epochs_final = int(modelCompileParameters[1])
        verbose_final = int(modelCompileParameters[3])
        batchsize = int(modelCompileParameters[2])
        learning_rate = float(modelCompileParameters[4])
        loss_function = modelCompileParameters[5]
        metrics_function = modelCompileParameters[6]
        optimizer_function =modelCompileParameters[7]

        # print(inputShape)
        # print(epochs_final)
        # print(batchsize)
        # print(learning_rate)
        # print(loss_function)
        # print(metrics_function)
        # print(optimizer_function)
        # print(arry1D[0][1])

        # with open('C:/code.txt', 'a') as append:
        #
        #     append.writelines("\n\t\tx = iris_data.data")
        #     append.writelines("\n\t\tiris_data.target.reshape(-1, 1)")
        #     append.writelines("\n\t\tencoder = OneHotEncoder(sparse=False)")
        #     append.writelines("\n\t\ty = encoder.fit_transform(y_)")
        #     append.writelines("\n\t\ttrain_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)")
        #     append.writelines("\n\t\tmodel = Sequential()")

        # x = data_set.iloc[:, :-1].values
        # y = data_set.iloc[:, 4].values

        y_ = data_set[modelCompileParameters[8]]
        x = data_set.drop(labels=[modelCompileParameters[8]],axis=1)

        y = array(y_)
        y_ = y.reshape(-1, 1)
        print("abjasdjajsadasd",y_.shape)
        # Convert data to a single column

        #One Hot encode the class labels
        encoder = OneHotEncoder(sparse=False)
        y = encoder.fit_transform(y_)
        print(y)


        # Split the data for training and testing
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=float(modelCompileParameters[10]))

        # Build the model
        model = Sequential()

        i=0

        with open('C:/code.txt', 'a') as append:
            append.writelines("\n\t\tx = dataset.data")

            while(i<count):
                if(i==0):
                    append.writelines("\n\t\tmodel.add(Dense("+arry1D[0][2]+"),input_shape=("+arry1D[0][1]+",),activation="+arry1D[0][1]+")")
                    model.add(Dense(int(arry1D[0][2]),input_shape=(inputShape,),activation=arry1D[0][1]))
                    print('Hello',model.add(Dense(int(arry1D[0][2]),input_shape=(inputShape,),activation=arry1D[0][1])))
                elif ((i+1)!=count):
                    append.writelines("\n\t\tmodel.add(Dense("+arry1D[i][2]+"),activation="+arry1D[i][1]+")")
                    model.add(Dense(int(arry1D[i][2]),activation=(arry1D[i][1])))
                elif ((i+1)==count):
                    append.writelines("\n\t\tmodel.add(Dense(" + arry1D[i][2] + "),activation=" + arry1D[i][1]+")")
                    model.add(Dense(int(arry1D[i][2]),activation=(arry1D[i][1])))
                i=i+1;


        # model.add(Dense(10, input_shape=(4,), activation='relu', name='fc1'))

        # model.add(Dense(10, activation='relu', name='fc2'))

        # model.add(Dense(3, activation='softmax', name='output'))

        # Adam optimizer with learning rate of 0.001


        if(optimizer_function=="Adam"):
            with open('C:/code.txt', 'a') as append:
                append.writelines("\n\t\toptimizer=Adam(lr="+str(learning_rate)+")")
            optimizer = Adam(lr=learning_rate)

        # elif ():
        #     optimizer = RMSrop(lr=learning_rate)

        with open('C:/code.txt', 'a') as append:
            append.writelines("\n\t\tmodel.compile("+str(optimizer)+",loss="+str(loss_function)+",metrics=["+str(metrics_function)+"])")
            append.writelines("\n\t\t#Train the model")
            append.writelines(
                "\n\t\thistory=model.fit(train_x,train_y,verbose=" + str(verbose_final) + ",batch_size=" + str(
                    batchsize) + ",epochs=" + str(epochs_final) + ",validation_data=(test_x,test_y)")
            append.writelines("\n")
            append.writelines("\t\t results = model.evaluate(test_x,test_y)")
            warnings.filterwarnings('always')


        model.compile(optimizer, loss=loss_function, metrics=[metrics_function])


        print('Neural Network Model Summary: ')

        print(model.summary())

        # with open('C:/code.txt', 'a') as append:
        #
        #     append.writelines("\n\t\t#Train the model")
        #     append.writelines("\n\t\thistory=model.fit(train_x,train_y,verbose="+str(verbose_final)+",batch_size="+str(batchsize)+",epochs="+str(epochs_final)+",validation_data=(test_x,test_y)")

        # Train the model
        from keras.callbacks import CSVLogger
        csv_logger = CSVLogger("c:/filesforproject/model_history_log.csv", append=True)

        history = model.fit(train_x, train_y, verbose=int(modelCompileParameters[3]), batch_size=batchsize, epochs=epochs_final, validation_data=(test_x, test_y),callbacks = [csv_logger])

        #y_pred = model.predict(test_x, verbose=0)
        #y_class = model.predict_classes(test_x, verbose=0)

        # y_pred = y_pred[:,0]
        # y_class = y_class[:,0]

        #f1 = f1_score(test_y, y_class)

        #print(f1)

        # Test on unseen data
        # with open('C:/code.txt', 'a') as append:
        #     append.writelines("\n")
        #     append.writelines("\t\tmodel.evaluate(test_x,test_y)")
        #     warnings.filterwarnings('always')
        #     append.writelines("\n")
        results = model.evaluate(test_x, test_y)

        #f1 = f1_score(test_y,results)

        print('Final test set loss: {:4f}'.format(results[0]))

        print('Final test set accuracy: {:4f}'.format(results[1]))

        with open('C:/code.txt', 'a') as append:
            append.writelines("\n\t\ty_pred = model.predict(test_x)")
            append.writelines("\n\t\trows = len(y_pred)")
            append.writelines("\n\t\tcols= len(y_pred[0])")
            append.writelines("\n\t\tfor i in range(rows):")
            append.writelines("\n\t\t\tfor j in range(cols):")
            append.writelines("\n\t\t\tfor j in range(cols):")
            append.writelines("\n\t\t\t\tif (y_pred[i][j] > 0.98):")
            append.writelines("\n\t\t\t\t\ty_pred[i][j] = 1")
            append.writelines("\n\t\t\t\telse:")
            append.writelines("\n\t\t\t\t\ty_pred[i][j] = 0")
            append.writelines("\n\t\t target_names = ['class 0', 'class 1', 'class 2']")
            append.writelines("\n\t\tloss_train = history.history['loss']")
            append.writelines("\n\t\tloss_val = history.history['val_loss']")
            append.writelines("\n\t\tepochs = range(1, epochs_final+1)")
            append.writelines("\n\t\tplt.plot(epochs, loss_train, 'g', label='Training loss')")
            append.writelines("\n\t\tplt.plot(epochs, loss_val, 'b', label='validation loss')")
            append.writelines("\n\t\tplt.title('Training and Validation loss')")
            append.writelines("\n\t\tplt.xlabel('Epochs')")
            append.writelines("\n\t\tplt.ylabel('Loss')")
            append.writelines("\n\t\tplt.legend()")
            append.writelines("\n\t\tplt.show()")

        y_pred = model.predict(test_x)
        rows = len(y_pred)
        cols= len(y_pred[0])
        for i in range(rows):

            for j in range(cols):
                if (y_pred[i][j] > 0.98):
                    y_pred[i][j] = 1
                else:
                    y_pred[i][j] = 0

        target_names = ['class 0', 'class 1', 'class 2']
        print(classification_report(test_y, y_pred, target_names=target_names))

        f1 = f1_score(test_y,y_pred,average=None)
        loss = results[0]
        acc = results[1]
        precision, recall, fscore, support = score(test_y, y_pred, average='macro')
        print('F1-Score - ', f1_score(test_y,y_pred,average=None))
        warnings.filterwarnings('always')

        loss_train = history.history['loss']
        loss_val = history.history['val_loss']
        epochs = range(1, epochs_final+1)  #epoch+1
        plt.plot(epochs, loss_train, 'g', label='Training loss')
        plt.plot(epochs, loss_val, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("c:/graphs/fig.png", dpi=100)
        plt.gcf()
        plt.show()
        plt.savefig("c:/graphs/fig.pdf",bbox_inches= "tight",pad_inches=2,transparent = True)

        global_list.append(fscore)
        global_list.append(acc)
        global_list.append(loss)
        global_list.append(precision)
        global_list.append(recall)
        global_list.append(support)
        return global_list

