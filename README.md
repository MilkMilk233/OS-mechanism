# CSC3150

CSC3150, 22 Fall

## Environment

Tencent Lighthouse, 2+ Core, 4 GB+ Mem, 60GB+ SSD, Ubuntu 20.04

## Prerequisites

1. https://cloud.tencent.com/developer/article/1061720 Setup guide for Lighthouse github configuration.

2. Remember to sync data into github, in case of data loss.

3. **Make image in keypoints，get ready to rollback，reinstall the system if necessary.**

## Download and install readily available kernel 

**2022/09/30: Ignore this session, DO NOT TRY IT HERE!!!!**  
**In this assignment we need to compile kernel by DIY.**  

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

## How to DIY(compile) a kernel?

### Preparations

First, **enter super user(root) mode**. This applys for all the following steps.  

```bash
sudo su
```

Install all dependencies
```bash

apt update && sudo apt install bc

apt-get install libncurses-dev gawk flex bison openssl libssl-dev dkms libelf-dev libudev-dev libpci-dev libiberty-dev autoconf llvm dwarves
```
Use `cd` to find a place where you want to store source file. Make sure you have enough permission. (For example, enter super user(root) by `sudo su`, and `cd ~/` to download file on path `/root/`)  
Download compressed package via wget
```bash
cd ~/
wget https://mirror.tuna.tsinghua.edu.cn/kernel/v5.x/linux-5.10.5.tar.xz
```
After download, the compressed package will be stored in the current folder.  
Unzip it with:
```bash
tar xvf linux-5.10.5.tar.xz
```
Then cd into the unzipped folder and execute:
```bash
cd ./linux-5.10.5/
make mrproper
make clean
```
Then download the config file to the same folder via wget
```bash
wget https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/.config
```

Then make sure you have a large-enough terminal window for the GUI of menuconfig. Enter:
```
make menuconfig
```
Then you get into a GUI. use four arrows to select, press enter to confirm.  
First select "Load" -> "OK".  
Back to homepage, select "Save" -> "OK".  
Back to homepage, select "Exit"  

Done. Now you are in command line terminal again.  

### Allocate memory to 'Swap Space'
[Some prior knowledge about Swap](https://www.cnblogs.com/ultranms/p/9254160.html)   

[Reference](https://cloud.tencent.com/developer/article/1704157)

All the commands should be executed in root mode as well.  

```bash
cd /usr     
mkdir swap      #create a new folder
dd if=/dev/zero of=/usr/swap/swapfile bs=1M count=4096 #Create a 4-GB memory space in SSD as virtual memory
du -sh /usr/swap/swapfile   #Check if this file occupy 4Gb
mkswap /usr/swap/swapfile
swapon /usr/swap/swapfile
```

Modify this file in vim.  
```
vim /etc/fstab
```

In vim, add this line at the end of the file: (Press `o` into insertion mode starting from next new line, press `Esc` then `:wq` to save and quit vim)  

```
/usr/swap/swapfile swap swap defaults 0 0
```

Then **reboot the machine**.
After reboot, you can check if the swap area is ready： 
```
free -m
```
If now Swap has ~4096 free space, Done!  

### Compile kernel(Choose either option)

#### Option 1: In terminal

Make sure you are in root mode.  

```bash
cd ~/linux-5.10.5/
make -j$(nproc)
```
It takes about 1~2 hrs to finish. Don't disconnect, don't close the terminal.  

#### Option 2: In process(Recommended)
[Reference](https://www.runoob.com/linux/linux-comm-nohup.html)  

Make sure you are in root mode.  

```bash
cd ~/linux-5.10.5/
nohup make -j$(nproc)       # Does not accept any intput, run in backend process, only can be killed by killing pid
```
It takes about 1~2 hrs to finish. You can leave for a coffee and close terminal. Don't worry.  
The command line output will be stored in ~/linux-5.10.5/nohup.out, use *vim* to inspect result.  
(Vim hints:`Shift + g` goes to the bottom of the file, press `:wq` to save and quit,`i` into insertion mode, `Esc` to quit,  `/` to search, after search press `enter` to locate)  

### Next steps

Make sure you are in root mode.  

```bash
cd ~/linux-5.10.5/
make modules_install
make install
```
Then reboot the machine.    
Enter this command to check kernel version.
```
uname -r
```
If you see 5.10.5, Now everything is done! 

### Recompile & Reinstall

Please start from the step 
```
make -j$(nproc)
make modules_install
make install
```
Then reboot.  
To save your time, don't start from `make mrproper`. 

## FAQ

### Cannot connect to VM via SSH after reinstall CVM (Third-person attack)

In your local machine, open `cmd`, enter
```
ssh-keygen -R 127.0.0.1
```
Replace `127.0.0.1` as your server IP address. 


## Assignment 1

### Program 1

#### Signal representations

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

### Program 2

#### Location of kernel func:
kernel_clone: /kernel/fork.c
do_execve: /fs/exec.c
do_wait: /kernel/exit.c
getname_kernel: /fs/namei.c

#### How to export?

After the function, add:
```
EXPORT_SYMBOL(FOO)
```

then recompile, re-install, reboot.  

Check this file to vertify if it's exposed:

~/linux-5.10.5/Module.symdvers


### 每次都要重新输密码？

试试在根目录下输入

```
git config --global credential.helper store
```
