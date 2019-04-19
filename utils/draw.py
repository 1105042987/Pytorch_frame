from re import findall
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')


def drawLossLine(Name, skipnrows=0, font_size=12,forward=False):
    with open('./logs/'+Name+'.log', 'r') as file:
        file.readline()
        onedata = file.readline()
        names = findall(r", (.+?):", onedata)
        data = np.array([findall(r": (.+?),", onedata)]).astype(float)
        onedata = file.readline()
        while onedata != '':
            data = np.append(data, np.array(
                [findall(r": (.+?),", onedata)]).astype(float), axis=0)
            onedata = file.readline()
    epoch = np.arange(len(data))
    if forward:
        data = data[:skipnrows, :]
        epoch = epoch[:skipnrows]
        suf = '_F'
    else:
        data = data[skipnrows:,:]
        epoch = epoch[skipnrows:]
        suf = ''
    for i, name in enumerate(names):
        plt.plot(epoch, data[:, i], label=name)
    plt.tick_params(labelsize=font_size)
    plt.legend(prop={'size': font_size}, loc="upper right")
    plt.savefig("./logs/fig/{}.png".format(Name+suf))
    plt.clf()
