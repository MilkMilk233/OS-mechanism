# CSC3150
CSC3150, 22 Fall

## Environment

Tencent Lighthouse, 2 Core 4 GB Mem, Ubuntu 20.04

## 配置虚拟机的方法

1. https://cloud.tencent.com/developer/article/1061720 按照方法配置腾讯云lighthouse github环境

2. 注意备份所有资料到github, 防止数据丢失

3. **提前做好镜像，随时准备回滚**

## Linux kernel & gcc 升级方法

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
### Nohup
[Ref](https://www.runoob.com/linux/linux-comm-nohup.html)

```bash
sudo su
nohup make -j$(nproc)
```

### Signal representations

- abort: exe 6
- 


