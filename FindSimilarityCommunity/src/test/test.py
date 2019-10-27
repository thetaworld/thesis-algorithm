import time

if __name__ == '__main__':
    start = time.time()
    for i in range(10000):
        print(i)
    print(time.time() - start)
