import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import seaborn as sn

plt.style.use('ggplot')


class KNN():

    def __init__(self):
        print("In Class Constructor")


    def knn(self,filepath,arry1D,modelCompileParameters,count):
        global_list = []
        file = open('C:/code.txt', "r+")
        file.truncate(0)
        file.close()


        with open('C:/code.txt', 'a') as append:
            append.writelines("\nimport matplotlib.pyplot as plt")
            append.writelines("\nfrom sklearn.metrics import accuracy_score, confusion_matrix")
            append.writelines("\nfrom sklearn.model_selection import train_test_split, cross_val_score")
            append.writelines("\nfrom sklearn.neighbors import KNeighborsClassifier")
            append.writelines("\nimport pandas as pd")
            append.writelines("\nimport seaborn as sn")
            append.writelines("\nplt.style.use('ggplot')")
            append.writelines("\n\nclass KNN():")
            append.writelines("\n\n\tdef __init__(self):")
            append.writelines("\n\tprint(""In Class Constructor"")")
            append.writelines("\n\n\tdef knn(self,filepath,arry1D,modelCompileParameters,count):")
            append.writelines("\n\n\t\tiris = pd.read_csv(""filepath"")")
            append.writelines("\n\t\ty = iris["+modelCompileParameters[8]+"]")
            append.writelines("\n\t\tX = iris.drop(labels=["+modelCompileParameters[8]+"],axis=1)")
            append.writelines("\n\t\tX_train, X_test, y_train, y_test = train_test_split(X, y, test_size="+modelCompileParameters[10]+")")
            append.writelines("\n\t\tknn = KNeighborsClassifier(n_neighbors="+arry1D[0][2]+")")
            append.writelines("\n\t\tknn.fit(X_train, y_train)")
            append.writelines("\n\t\ty_pred = knn.predict(X_test)")
            append.writelines("\n\t\tcm = confusion_matrix(y_test, y_pred)")
            append.writelines("\n\t\taccuracy = accuracy_score(y_test, y_pred)*100")
            append.writelines("\n\t\tprint('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')")


        trainData = pd.read_csv(filepath)

        y = trainData[modelCompileParameters[8]]
        X = trainData.drop(labels=[modelCompileParameters[8]], axis=1)


        # y = trainData["Species"]
        # X = trainData.drop(labels=["Species"], axis=1)

        # Split arrays or matrices into random train and test subsets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(modelCompileParameters[10]))

        # Create KNN Classifier
        # Number of neighbors to use by default for kneighbors queries.
        knn = KNeighborsClassifier(n_neighbors=int(arry1D[0][2]))
        # Train the model using the training sets
        knn.fit(X_train, y_train)
        # Predict the response for test dataset
        print("Response for test dataset:")
        y_pred = knn.predict(X_test)
        print(y_pred)

        conf_metrics = confusion_matrix(y_test, y_pred)

        print(conf_metrics)

        accuracy = accuracy_score(y_test, y_pred) * 100
        print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')

        list_range = arry1D[0][3].split(",")
        # creating list of K for KNN
        k_list = list(range(int(list_range[0]),int(list_range[1]),int(list_range[2])))
        # creating list of cv scores
        cv_scores = []
        with open('C:/code.txt', 'a') as append:
            append.writelines("\n\t\tk_list = list(range("+list_range[0]+","+list_range[1]+","+list_range[2]+"))")
            append.writelines("\n\t\tcv_scores = []")
            append.writelines("\n\t\tfor k in k_list:")
            append.writelines("\n\t\t\tknn = KNeighborsClassifier(n_neighbors=k)")
            append.writelines("\n\t\t\tscores = cross_val_score(knn, X_train, y_train, cv=int(arry1D[0][4]), scoring='accuracy')")
            append.writelines("\n\t\t\tcv_scores.append(scores.mean())")
            append.writelines("\n\n\t\tMSE = [1 - x for x in cv_scores]")
            append.writelines("\n\n\t\tplt.figure()")
            append.writelines("\n\t\tplt.figure(figsize=(15, 10))")
            append.writelines("\n\t\tplt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')")
            append.writelines("\n\t\tplt.xlabel('Number of Neighbors K', fontsize=15)")
            append.writelines("\n\t\tplt.ylabel('Misclassification Error', fontsize=15)")
            append.writelines("\n\t\tsn.set_style(""whitegrid"")")
            append.writelines("\n\t\tplt.plot(k_list, MSE)")
            append.writelines("\n\t\tplt.show()")
            append.writelines("\n\t\tbest_k = k_list[MSE.index(min(MSE))]")
            append.writelines("\n\t\tprint(""The optimal number of neighbors is %d."" % best_k)")

        # perform 10-fold cross validation
        for k in k_list:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_train, y_train, cv=int(arry1D[0][4]), scoring=modelCompileParameters[6])
            cv_scores.append(scores.mean())

        MSE = [1 - x for x in cv_scores]

        plt.figure()
        plt.figure(figsize=(15, 10))
        plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
        plt.xlabel('Number of Neighbors K', fontsize=15)
        plt.ylabel('Misclassification Error', fontsize=15)
        sn.set_style("whitegrid")
        plt.plot(k_list, MSE)
        plt.savefig("c:/graphs/fig.png", dpi=100)
        plt.gcf()
        plt.show()
        plt.savefig("c:/graphs/fig.pdf", bbox_inches="tight", pad_inches=2, transparent=True)

        best_k = k_list[MSE.index(min(MSE))]
        print("The optimal number of neighbors is %d." % best_k)


        global_list.append(conf_metrics)
        global_list.append(str(round(accuracy, 2)) + ' %.')
        global_list.append(best_k)
        return global_list