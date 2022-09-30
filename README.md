# CSC3150
CSC3150, 22 Fall

## Environment

Tencent Lighthouse, 2 Core 4 GB Mem, Ubuntu 20.04

## 配置虚拟机的方法

1. https://cloud.tencent.com/developer/article/1061720 按照方法配置腾讯云lighthouse github环境

2. 注意备份所有资料到github, 防止数据丢失

3. **提前做好镜像，随时准备回滚**

## Linux kernel & gcc 升级方法 (Ignore in this assignment, not adaptable)

Normal Update:

```bash

wget https://raw.githubusercontent.com/pimlie/ubuntu-mainline-kernel.sh/master/ubuntu-mainline-kernel.sh

sudo install ubuntu-mainline-kernel.sh /usr/local/bin/

sudo ubuntu-mainline-kernel.sh -i 5.10.10   #To update kernel version to 5.10.10

```

Then restart your VM.

```bash

uname –r    #Check linux kernel = 5.10.10

gcc -v      #Check gcc > 4.9

```

## How to compile kernel?

### Preparations

Install all dependencies
```bash
sudo apt-get install libncurses-dev gawk flex bison openssl libssl-dev dkms libelf-dev libudev-dev libpci-dev libiberty-dev autoconf llvm dwarves
```
Use `cd` to a place where you want to store source file. Make sure you have enough permission. (For example, `cd ~/`)

Download compressed package via wget
```bash
wget https://mirror.tuna.tsinghua.edu.cn/kernel/v5.x/linux-5.10.5.tar.xz
```
After download, it will be stored in the current folder.

Unzip it with
```bash
sudo tar xvf linux-5.10.5.tar.xz
```
then cd to the folder
```bash
cd /boot
ls
```
with ls, you can find a lot of files with prefix 'config'. Select one of them, copy it to the 'linux-5.10.5' folder you just unzip, rename it as '.config'
For example:
```bash
cp /boot/config-5.4.0-121-genetic ~/linux-5.10.5/.config
```
then use cd back to the folder 
```bash
cd ~/linux-5.10.5/
```
then make sure you have a large-enough terminal window for the GUI of menuconfig. Enter:
```
make menuconfig
```
Then you get into a GUI. use four arrows to select and press enter to confirm.  
First select "Load" -> "OK".  
Back to homepage, select "Save" -> "OK".
Back to homepage, select "Exit"  

Done. Now you are in command line terminal again.  

### Allocate memory
[Some prior knowledge about Swap](https://www.cnblogs.com/ultranms/p/9254160.html)   

[Reference](https://cloud.tencent.com/developer/article/1704157)

```bash
cd /usr     
mkdir swap      #create a new folder
sudo dd if=/dev/zero of=/usr/swap/swapfile bs=1M count=4096 #Create a 4-GB memory space fo;e
sudo du -sh /usr/swap/swapfile   #Check if this file occupy 4Gb
sudo mkswap /usr/swap/swapfile
sudo swapon /usr/swap/swapfile
sudo vim /etc/fstab
```

in vim, add this line as attachment below:

```
/usr/swap/swapfile swap swap defaults 0 0
```

Then reboot the machine.
Check if the swap area is ready：

```
free -m
```
Done!

### Nohup
[Ref](https://www.runoob.com/linux/linux-comm-nohup.html)

```bash
sudo su
nohup make -j$(nproc)
```

### Next steps

```
sudo su
make modules_install
make install
```
Then reboot the machine.  
Done!


### Signal representations

- abort: exe 6
- alarm: exe 14
- bus: exe 7
- floating: exe 8
- hangup: exe 1
- illegal_instr: exe 4
- ubterrupt: exe 2
- kill: exe 9
- Normal
- pipe: exe 13
- quit: exe 3
- segment_fault: exe 11
- stop: stop 19
- terminate: exe 15
- trap: exe 5


