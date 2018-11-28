
import numpy as np

path = 'data/fridgeb4-labels'
# path2 = '1ydata/washing machine1'
f = open(path)
# f2 = open(path2)
#
# flag = False
# for l1,l2 in zip(f, f2):
#     if l1 != l2:
#         flag = True; break
#
# if flag: print("files different")
count = 0
for line in f:
    count+=1

print(count)


