#Thrust Bug
----------
This repo exposes a bug present with PGI OpenACC and thrust interoperability. thrust.cu contains an extern "C" defined function, `CopyIfLessThanOrEqual`, which calls a thrust function to copy elements of the input array less than the specified value, this thrust function is executed in the provided stream. When OpenACC is used to create the input array and the thrust function is called twice the second thrust return value is incorrect. When thrust is used to create the input array and a cuda stream is explicitly created the results are correct.

##Build
```
$ make
$ ./acc.out
num copied: 500000 500000
$ ./stream.out
copied: 500000, 1000000
```
