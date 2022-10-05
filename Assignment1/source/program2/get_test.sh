make clean
make
gcc -o test test.c
insmod program2.ko
rmmod program2
dmesg | grep program2
