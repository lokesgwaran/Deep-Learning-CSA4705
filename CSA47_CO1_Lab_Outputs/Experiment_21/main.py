import numpy as np
a=np.float32(.1);b=np.float64(.1);print('float32 representation:',format(a,'.20f'));print('float64 representation:',format(b,'.20f'))
x32=np.float32(0);x64=np.float64(0)
for i in range(100000):x32+=np.float32(.0001);x64+=np.float64(.0001)
print('float32 sum:',x32);print('float64 sum:',x64);print('Expected sum: 10.0')
large32=np.float32(100000);large64=np.float64(100000);print('float32 subtraction:',(large32+np.float32(.01))-large32);print('float64 subtraction:',(large64+np.float64(.01))-large64)
