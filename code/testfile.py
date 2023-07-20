path="/Users/liubo22/Downloads/TextPair/TextCNN/result3.txt"
import random

dic=dict()
with open(path) as f:
    for line in f:
        # print(line)
        try:
            if '4467987623121878' in line:
                print(line)
            if "发布了头条文章" in line.split("\t")[1] and '秃' in line.split("\t")[1]:
            #     p = random.randint(0, 100)
            #     p *= 0.000001
                dic[line.split("\t")[0] + "\t"+line.split("\t")[1] + "\t" + line.split("\t")[3].split('+')[0]+"\t"] = str(float(line.split("\t")[2].split("+")[0])+0.10)

            elif "发布了头条文章" in line.split("\t")[1] and '银行卡' in line.split("\t")[1]:
                dic[line.split("\t")[0] + "\t" + line.split("\t")[1] + "\t"  +
                    '银行卡'] = str(float(line.split("\t")[2].split("+")[0]) + 0.13)
            elif '拿肉' in line.split("\t")[1]:
                dic[line.split("\t")[0] + "\t" + line.split("\t")[1] + "\t" + line.split("\t")[3].split('+')[0] + "\t"
                    ] = str(float(line.split("\t")[2].split("+")[0]) + 0.10)
            elif '查询开房记录' in line.split("\t")[1] or '查开房记录' in line.split("\t")[1]:
                dic[line.split("\t")[0] + "\t" + line.split("\t")[1] + "\t" + line.split("\t")[3].split('+')[0] + "\t"
                    ] = str(float(line.split("\t")[2].split("+")[0]) + 0.06)
            else:
                if ' ' in line.split('\t')[3] or '信用卡' in line.split('\t')[3]:
                    dic[line.split("\t")[0] + "\t" + line.split("\t")[1] + "\t" + line.split("\t")[4].split('+')[
                        0] + "\t"] = line.split("\t")[2].split("+")[0]
                else:
                     dic[line.split("\t")[0] + "\t"+line.split("\t")[1]+"\t"+line.split("\t")[3].split('+')[0]+"\t"]=line.split("\t")[2].split("+")[0]
        except IndexError as e:
            pass
        except ValueError as e:
            pass

kedict = sorted(dic.items(), key=lambda x: x[1], reverse=True)
# print(kedict)


wpath='output33.txt'
count=0

# for item in kedict:
#     # if "秃" in item[0]:
#         count+=1
#         if count<7000:
#             print(item[0]+"=="+item[1])

print(count)
with open(wpath,"w") as fp:
    for item in kedict:
        # if float(item[1])>0.9:
        if '@秃' not in item[0]:
        # if "四件套" not in item[0] and "支付宝" not in item[0] and "秃头" not in item[0] and "大额" not in item[0] and "安娜" not in item[0] and "叶子" not in item[0]:

                 fp.write(item[0].replace('\n','')+"\t"+item[1]+"\n")
