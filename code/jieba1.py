path='/Users/liubo22/Desktop/mid.txt'

with open(path) as fp:
    for line in fp:
        line1=line.split('/')
        mid=line1[-1].replace('\n','').split('?')
        print(mid[0])