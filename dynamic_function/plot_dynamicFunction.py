import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

x = np.arange(-6, 6, 0.05) #x軸の描画範囲の生成。0から10まで0.05刻み。
y = np.arange(-6, 6, 0.05) #y軸の描画範囲の生成。0から10まで0.05刻み。

def plot(data):
    plt.cla()                      # 現在描写されているグラフを消去
        # 100個の乱数を生成
    im = plt.plot(rand)            # グラフを生成

ani = animation.FuncAnimation(fig, plot, interval=100)
plt.show()
