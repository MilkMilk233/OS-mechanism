clear
make clean
make
sleep 1
gcc -o test test.c
insmod program2.ko
sleep 1
rmmod program2
sleep 1
dmesg | grep program2

