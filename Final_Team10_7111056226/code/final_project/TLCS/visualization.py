import matplotlib.pyplot as plt
import os

class Visualization:
    def __init__(self, path, dpi):
        self.path = path
        self.dpi = dpi

    def save_data_and_plot(self, data, filename, xlabel, ylabel):
        min_val = min(data)
        max_val = max(data)
        plt.rcParams.update({'font.size': 24})

        # 創建圖表
        plt.plot(data)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.margins(0)
        plt.ylim(min_val - 0.1 * abs(min_val), max_val + 0.1 * abs(max_val))

        # 設置圖表大小並將其保存為 png 圖像
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self.path, filename+'.png'), dpi=self.dpi)
        plt.close("all")

        with open(os.path.join(self.path, filename + '_data.txt'), "w") as file:
            for value in data:
                file.write("%s\n" % value)
    