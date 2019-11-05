def read_write_line(path_from, path_to):
    ms = open(path_from)
    for line in ms.readlines():
        line = line.strip().split(" ")[0:2]
        line = str(line[0])+" "+str(line[1])+"\n"
        print(line)
        with open(path_to, "a") as mon:
            mon.write(line)


if __name__ == '__main__':
    read_write_line(b"D:\workspace\pycharm\paper_algorithm\FindSimilarityCommunity\src\data\preprocessData\M1.edges",
                    b"D:\workspace\pycharm\paper_algorithm\FindSimilarityCommunity\src\data\preprocessData\M1_1.edges")