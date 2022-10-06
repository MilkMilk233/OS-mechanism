cd /tmp/
rm test
cd /home/vagrant/CSC3150/Assignment1/source/program2
gcc -o test test.c
cp test /tmp/
clear
make clean
make
insmod program2.ko
sleep 1
rmmod program2
dmesg | grep program2


