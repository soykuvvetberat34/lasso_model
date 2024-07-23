from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso ,LassoCV
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score,mean_squared_error

datas=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\Hitters.csv")
datas=datas.dropna()
dms=pd.get_dummies(datas[["League","Division","NewLeague"]])
y=datas["Salary"]
x_=datas.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
x=pd.concat([x_,dms],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=99)

lasso_model=Lasso()
lasso_model.fit(x_train,y_train)
lambdalar2=10**(np.linspace(-4,4,100))

#lasso cross validation model ile optimum alpha bulma
lassoCV=LassoCV(alphas=lambdalar2,cv=10,max_iter=100000).fit(x_train,y_train)
opt_alp=lassoCV.alpha_
lasso_tuned = Lasso(alpha=opt_alp).fit(x_train, y_train)
predict=lasso_tuned.predict(x_test)
RMSE=np.sqrt(mean_squared_error(y_test,predict))
print(RMSE)

series=pd.Series(lasso_tuned.coef_,index=x_train.columns)
print(series)#bu dizideki 0 olan değerler gereksiz(anlamsız) değerlerdir
#diğer değerlerse hedef değişkene etkileridir



