import PyQt5
from PyQt5.QtCore import *

def compree(file,n):
    f = open(file,'rb')
    data = f.read()
    return qCompress(data,n)

a = compree('/home/tt/tt.txt',100)

f = open('/home/tt/tx.txt','wb')
f.write(a)

print(a)
