from matplotlib import pyplot as plt
import numpy as np

def plot_params():
    file = open("delta_a.txt","r")
    data = []

    lines = file.readlines()
    for line in lines:
        data.append(line.replace("\n","").split(";"))

    x = []
    y = []

    for r in data:
        x.append(float(r[0]))
        y.append(float(r[3]))

    plt.xlabel('kąt obrotu układu')
    plt.ylabel('średni błąd średniokwadratowy')
    plt.title('Średni błąd średniokwadratowa w zależności od kąta obrotu układu')
    plt.plot(x,y)
    plt.show()

    file.close()

def plot_masks_error():
    file = open("masks.txt","r")
    data = []
    x = []
    y = []

    lines = file.readlines()
    for line in lines:
        data.append(line.replace("\n","").split(";"))

    for r in data:
        x.append(str(r[0]))
        y.append(float(r[1]))

    print(x,y)

    fig, ax = plt.subplots()
    index = np.arange(len(lines))

    rects1 = ax.bar(index, y, 0.35,alpha=0.4, color='b')
    
    ax.set_xlabel('maski')
    ax.set_ylabel('średni błąd średniokwadratowy')
    ax.set_title('Średni błąd średniokwadratowy w zależności od maski')
    # ax.set_xticklabels(x)
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x, rotation = 0)
    ax.set_ylim((1100, 1300))
    fig.tight_layout()
    plt.show()

    file.close()

plot_masks_error()