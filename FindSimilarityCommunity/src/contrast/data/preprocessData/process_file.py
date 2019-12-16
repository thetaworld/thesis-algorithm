import numpy as np


def read_write_line(path_from, path_to):
    ms = open(path_from)
    for line in ms.readlines():
        line = line.strip().split(" ")[0:2]
        line = str(line[0])+" "+str(line[1])+"\n"
        print(line)
        with open(path_to, "a") as mon:
            mon.write(line)


def produce_text_info(path_to):
    a = np.random.randint(0, 2, size=(2405, 10))
    for i, line in enumerate(a):
        with open(path_to, "a") as mon:
            mon.write(str(i)+" "+str(line)[1:-1]+"\n")


if __name__ == '__main__':
    # read_write_line(b"D:\workspace\pycharm\paper_algorithm\FindSimilarityCommunity\src\data\preprocessData\M1.edges",
    #                 b"D:\workspace\pycharm\paper_algorithm\FindSimilarityCommunity\src\data\preprocessData\M1_1.edges")
    produce_text_info(b"D:\workspace\pycharm\paper_algorithm\FindSimilarityCommunity\src\data\preprocessData\info_2045")