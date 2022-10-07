cd /tmp/
rm test
cd /home/vagrant/CSC3150/Assignment1/source/program2
gcc -o test test.c
cp test /tmp/
clear
make clean
make
sleep 5
insmod program2.ko
sleep 5
rmmod program2
dmesg | grep program2


