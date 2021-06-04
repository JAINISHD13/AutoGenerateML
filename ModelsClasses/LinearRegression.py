from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score


class LinearRegression():


    def linearRegression(self,filepath,arry1D,modelCompileParameters,count):
        global_list = []

        file = open('C:/code.txt', "r+")
        file.truncate(0)
        file.close()

        with open('C:/code.txt', 'a') as append:
            append.writelines("\nfrom sklearn.model_selection import train_test_split")
            append.writelines("\nimport pandas as pd")
            append.writelines("\nimport numpy as np")
            append.writelines("\nfrom sklearn import linear_model")
            append.writelines("\nfrom sklearn.metrics import r2_score")
            append.writelines("\n\nclass LinearRegression():")
            append.writelines("\n\n\tdef linearRegression(self,filepath,arry1D,modelCompileParameters,count):")
            append.writelines("\n\t\tdf_train = pd.read_csv(filepath)")
            append.writelines("\n\t\tx = df_train['x']")
            append.writelines("\n\t\ty = df_train['y']")
            append.writelines("\n\t\tx = np.array(x)")
            append.writelines("\n\t\ty = np.array(y)")
            append.writelines("\n\t\tx = x.reshape(-1, 1)")
            append.writelines("\n\t\tx_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float("+modelCompileParameters[10]+"))")
            append.writelines("\n\t\tprint(x_train)")
            append.writelines("\n\t\tclf = linear_model.LinearRegression()")
            append.writelines("\n\t\tclf.fit(x_train, y_train)")
            append.writelines("\n\t\ty_pred = clf.predict(x_test)")
            append.writelines("\n\t\tprint(r2_score(y_test, y_pred))")

        df_train = pd.read_csv(filepath)
        x = df_train['x']
        y = df_train['y']
        # x_test = df_test['x']
        # y_test = df_test['y']

        x = np.array(x)
        y = np.array(y)
        # x_test = np.array(x_test)
        # y_test = np.array(y_test)

        x = x.reshape(-1, 1)
        # x_test = x_test.reshape(-1,1)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(modelCompileParameters[10]))
        print(x_train)

        clf = linear_model.LinearRegression()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print(r2_score(y_test, y_pred))

        global_list.append(r2_score(y_test,y_pred))

        return  global_list
