# CSC3150 Assignment 2
Chen Zhixin, 120090222

## Main Part

### Prerequisites 

Environment: Tencent Lighthouse, Ubuntu 20.04, Kernel Version 5.10.x

Check g++ version > 4.9  
```
g++ --version
```
Check kernel version ~5.10.x
```
uname -r
```
Install `libncurses5-dev`
```bash
sudo apt-get install libncurses5-dev
```

### How to compile and run the program?

```
cd ~/CSC3150/Assignment2/source
g++ hw2.cpp -lpthread
./a.out
```

## Bonus Part

思路：开一个queue用来装任务，然后所有的线程轮流提请任务列表。