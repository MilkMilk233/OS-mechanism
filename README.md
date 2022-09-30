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
[Reference](https://blog.csdn.net/babybabyup/article/details/79815118)

```bash
free -m     #To check memory status
dd if=/dev/zero of=/swapfile bs=1k count=1024000    # Allocate 1GB to Swap
mkswap /swapfile
swapon /swapfile
```

### Nohup
[Ref](https://www.runoob.com/linux/linux-comm-nohup.html)

```bash
sudo su
nohup make -j$(nproc)
```


