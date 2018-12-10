import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-20, 20, 0.05) #x軸の描画範囲の生成。0から10まで0.05刻み。
y = np.arange(-20, 20, 0.05) #y軸の描画範囲の生成。0から10まで0.05刻み。

X, Y = np.meshgrid(x, y)

Z = np.power((np.power(X,2) + Y - 11),2) + np.power((X + np.power(Y,2) - 7),2)   # 表示する計算式の指定。等高線はZに対して作られる。
#Z = (1 - 1/(1 + 0.05 * (np.power(X,2)+(np.power((Y - 10),2)))) - 1/(1 + 0.05*(np.power((X - 10),2) + np.power(Y,2))) - 1/(1 + 0.03*(np.power((X + 10),2) + np.power(Y,2))) - 1/(1 + 0.05*(np.power((X - 5),2) + np.power((Y + 10),2))) - 1/(1 + 0.1*(np.power((X + 5),2) + np.power((Y + 10),2))))*(1 + 0.0001*np.power((np.power(X,2) + np.power(Y,2)),1.2))
#plt.pcolormesh(X, Y, Z, cmap='hsv') # 等高線図の生成。cmapで色付けの規則を指定する。
plt.pcolormesh(X, Y, Z,cmap='hsv') # 等高線図の生成。cmapで色付けの規則を指定する。
plt.colorbar()
#pp=plt.colorbar (orientation="vertical") # カラーバーの表示
#pp.set_label("Label", fontname="Arial", fontsize=24) #カラーバーのラベル

plt.xlabel('X', fontsize=24)
plt.ylabel('Y', fontsize=24)

plt.show()

#Z = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
