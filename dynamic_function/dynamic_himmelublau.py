import numpy as np
import matplotlib.pyplot as plt


x = np.arange(-5, 5, 0.05) #x軸の描画範囲の生成。0から10まで0.05刻み。
y = np.arange(-5, 5, 0.05) #y軸の描画範囲の生成。0から10まで0.05刻み。

X, Y = np.meshgrid(x, y)

def map(gen):
    if gen <= 22:
        num = gen
    elif 22 < gen <= 44:
        num = 44 - gen
    elif 44 < gen <= 66:
        num = gen - 45
    elif 66 < gen <= 88:
        num = 88 - gen
    elif 88 < gen:
        num = gen - 89

    return num

for gen in range(100):
    Z = np.power((np.power(X,2) + Y - map(gen)),2) + np.power((X + np.power(Y,2) - map(gen)),2)   # 表示する計算式の指定。等高線はZに対して作られる。
    #Z = (1 - 1/(1 + 0.05 * (np.power(X,2)+(np.power((Y - 10),2)))) - 1/(1 + 0.05*(np.power((X - 10),2) + np.power(Y,2))) - 1/(1 + 0.03*(np.power((X + 10),2) + np.power(Y,2))) - 1/(1 + 0.05*(np.power((X - 5),2) + np.power((Y + 10),2))) - 1/(1 + 0.1*(np.power((X + 5),2) + np.power((Y + 10),2))))*(1 + 0.0001*np.power((np.power(X,2) + np.power(Y,2)),1.2))
    #plt.pcolormesh(X, Y, Z, cmap='hsv') # 等高線図の生成。cmapで色付けの規則を指定する。
    plt.pcolormesh(X, Y, Z,cmap='hsv') # 等高線図の生成。cmapで色付けの規則を指定す
    #pp=plt.colorbar (orientation="vertical") # カラーバーの表示
    #pp.set_label("Label", fontname="Arial", fontsize=24) #カラーバーのラベル

    plt.xlabel('X', fontsize=24)
    plt.ylabel('Y', fontsize=24)

    if len(str(gen))==1:
        plt.savefig('dynamic_himmelblau/00'+str(gen)+'.png')
    elif(len(str(gen))) == 2:
        plt.savefig('dynamic_himmelblau/0'+str(gen)+'.png')
    elif(len(str(gen))) == 3:
        plt.savefig('dynamic_himmelblau/'+str(gen)+'.png')
    #savefig('cma_es_pic/figure'+str(gen)+'.png')
    plt.clf()
