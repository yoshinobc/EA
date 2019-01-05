import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5.05, 0.05) #x軸の描画範囲の生成。0から10まで0.05刻み。
y = np.arange(-5, 5.05, 0.05) #y軸の描画範囲の生成。0から10まで0.05刻み。

X ,Y= np.meshgrid(x, y)

c1 = -2 * np.ones((2,201,201))
c2 = 4 * np.ones((2,201,201))
print(X)
def func(lists):
    return (1 - 1 / (1 * (np.array(lists) - c1) + 1)) + (1 - 1 / (2 * (np.array(lists) - c2) + 1))

Z = (1 - 1 / (1 * np.linalg.norm(np.array([X,Y]) - c1, axis=0) + 1)) + (1 - 1 / (2 *np.linalg.norm(np.array([X,Y]) - c2,  axis=0) + 1))

print(len(Z))
#Z = (1 - 1 / (1 * np.linalg.norm(np.array([X,Y]) - c1) + 1)) + (1 - 1 / (2 * np.linalg.norm(np.array([X,Y]) - c2) + 1))
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
"""
z = np.zeros([200,200])
for i in range(200):
    for j in range(200):
        _x = x[i]
        _y = y[j]
        z[i,j] =  (1 - 1 / (1 * np.linalg.norm(np.array([_x,_y]) - np.array([-2,-2])) + 1)) + (1 - 1 / (2 * np.linalg.norm(np.array([_x,_y]) - np.array([4,4])) + 1))


plt.imshow(z,cmap="gist_rainbow")
plt.colorbar () # カラーバーの表示
plt.xlabel('X')
plt.ylabel('Y')
plt.xticks([-5,5])
plt.yticks([-5,5])
plt.show()
"""
