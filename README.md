# CSC3150

CSC3150, 22 Fall

## Environment

Tencent Lighthouse, 2+ Core, 4 GB+ Mem, 60GB+ SSD, Ubuntu 20.04

## 配置虚拟机的方法

1. https://cloud.tencent.com/developer/article/1061720 按照方法配置腾讯云lighthouse github环境

2. 注意备份所有资料到github, 防止数据丢失

3. **提前做好镜像，随时准备回滚，必要时可重装系统**

## Linux kernel & gcc 升级方法 (Ignore in this assignment, DO NOT TRY IT HERE!!!!)

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
cd ~/
wget https://mirror.tuna.tsinghua.edu.cn/kernel/v5.x/linux-5.10.5.tar.xz
```
After download, the compressed package will be stored in the current folder.  
Unzip it with:
```bash
sudo tar xvf linux-5.10.5.tar.xz
```
Then cd into the unzipped folder and execute:
```bash
cd ./linux-5.10.5/
sudo make mrproper
sudo make clean
```
then download the config file via wget
```bash
wget https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/.config
```

then make sure you have a large-enough terminal window for the GUI of menuconfig. Enter:
```
sudo su
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

in vim, add this line at the end of the file: (Press `i` into insertion mode, press `Esc` then `:wq` to save and quit vim)  

```
/usr/swap/swapfile swap swap defaults 0 0
```

Then **reboot the machine**.
After reboot, you can check if the swap area is ready：
```
free -m
```
If you see Swap has ~4096 avaliable, Done!

### Compile kernel(Choose either option)

#### Option 1: in terminal

```bash
cd ~/linux-5.10.5/
sudo su
make -j$(nproc)
```
It takes about 1~2 hrs to finish. Don't disconnect, don't close the terminal.
If ends properly, the final lines should be:
```bash

```

#### Option 2:In process
[Ref](https://www.runoob.com/linux/linux-comm-nohup.html)

```bash
cd ~/linux-5.10.5/
sudo su
nohup make -j$(nproc)       # Does not accept any intput, run in backend process, only can be killed by killing pid
```
It takes about 1~2 hrs to finish. You can leave for a coffee and close terminal. Don't worry.
The command line output will be stored in ~/linux-5.10.5/nohup.out, use *vim* to inspect result.
(Vim hints:`Shift + g` goes to the bottom of the file, press `:wq` to save and quit,`i` into insertion mode, `Esc` to quit,  `/` to search, after search press `enter` to locate)

### Next steps

```bash
cd ~/linux-5.10.5/
sudo su
make modules_install
make install
```
Then reboot the machine.  
Now everything is done!


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


