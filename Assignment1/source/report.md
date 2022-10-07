# CSC3150 Assignment 1 Report

Chen Zhixin, 120090222

[TOC]

## How I design my program?

### Program 1

In program 1, I implemented a fork function in user mode. By referring to tutorial slides, I gained a basic understanding about how process works, and how process in user mode interact with the functions in kernel. The task is to clone a child process, let it run an executable and return a signal to the parent process. As for the parent process, all it need to do is to wait for the signal, then continue and display the signal received. There are three key functions that we need to utilize:

1. `fork()`: To folk a child process.
2. `execve()`: To run a executable with indicated path.
3. `waitpid()`: To wait for a specific process's termination, and receive signal *(raised by `raise()` in executable)* from it.

First, use `fork()` to folk a child process, then both the parent and child process will start executing at the next line, the only difference is the return value `pid` will be different, 0 for child process, non-zero integer for parent process. Then we use this to distinguish parent and child process, let them act differently.

Then, for the parent process, it uses `waitpid()` function to wait for the child's response. This function is blocking, i.e., It will be released only after it receives any signal from child process. Then, with the signal received, It can determine which type of signal is and display it to the user.

As for the child process, first it read the arguments passed in the command line, use it as the path of the executable, then run it with `execve()`. If the function run properly, it will never return and replace the original resource with the new executable. Then in the new one, signals are raised, received by the original parent process. Then it circle around, back to previous parent's waiting step and continue.

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig1.jpg" alt="fig1" style="zoom: 50%;" />

### Program 2

In program 2, I implemented a fork function in kernel mode. Similar to program 1, this time we need to fork a process, but not in user space, it's in kernel instead. Kernel environment is a little bit different than the user one. First, the way of running it is different. We need to utilize the "LKM" - or "loadable kernel modules" mechanism to load our own-written kernel codes. After compile, we need to insert the module by `insmod program2.ko`. Then the module get initialized, triggering the functions in this program.

After kernel module initializes, a kernel thread is created, in order to activate functions work in kernel space. `kthread_create()` is needed for creating a "task". And the `wake_up_process()` wake this "task" up, the function eventually runs in kernel space.

There are several key functions to be utilized in this program. 

1. `do_execve()`: similar to `execve()`, but in kernel version.
2. `do_wait()`: similar to `wait()`, but in kernel version
3. `kernel_clone()`: a little bit similar to `fork()`, but in kernel version, also need to specify function to execute.
4. `getname_kernel()`: convert name in string to "filename" structure.

These functions are defined in the source file of the Linux kernel. Some of them are defined as "static", but some are not. For non-static part (`kernel_clone()` and `getname_kernel()` ), we need to add `extern` on that function, and attach `EXPORT_SYMBOL(function())` below. Details will be illustrated in below parts.

For function design, I separate into three functions:

- `my_fork()`:
- `my_wait()`:
- `my_exec()`:

After the kernel thread is created, `my_fork()` function is under execution. By referring to the source code, to clone a process, we need to use `kernel_clone()`. It needs some arguments, whose structure `kernel_clone_args()`is defined in the [source code](https://elixir.bootlin.com/linux/v5.10.5/source/include/linux/sched/task.h#L21). Then we create arguments to pass in:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig3.jpg" alt="fig3" style="zoom:50%;" />

Some key points: 

- .flags set to "SIGCHLD"
- .exit_signal set to "SIGCHLD"
- .stack set to "(unsigned long) & my_Exec"
- Others set to 0 or NULL, depending on it is integer or pointer.

Then we pass in the arguments. During the execution of `kernel_clone()`, actually a child process is created, running the `my_exec()` function, which will be introduced later. Then the parent process will print out the PID of both process, and wait for child process's termination by function `my_wait()`. the `my_wait()` function will return a value `status`, whose last 7 bits are exactly the standardized SIGNAL that can be analysis by us manually (We use and operation with 0x7f to get real status).

As for the `my_exec()` function run by child process, it is a bit similar to the function `execve()` in user mode.  Here we indicate a string path `/tmp/test`, using function `getname_kernel()` convert it to structure "filename", latter can be passed into `do_execve()`, which runs a executable in user space and return value to be received in parent process.

Meanwhile, parent process get into wait status with the function `my_wait()`. This function also needs a special structure called "wait_ops". We define it like this:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig5.jpg" alt="fig5" style="zoom:50%;" />

Notice that in kernel version 5.10.x, the type of `.wo_stat` has change from `int*` to `int`. So here we use an integer to initialize it. Then we pass it into the `do_execve()` function. After it receive signal from child process and release, we will get the signal back in `wo.wo_stat` and return it back to the main function. Then we utilize this to get signal, continue. The general routine is like the figure below:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig2.jpg" style="zoom:50%;" />

### Bonus

In bonus, I implemented a pstree function.



## How to set up development environment?

### Prerequisites

**Make image in key points，get ready to rollback，reinstall the system if necessary.**

### Preparations

First, **enter super user(root) mode**. This applies for all the following steps.  

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
wget https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/.config		# Expire date: 2022/3/16
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

### Allocate memory to 'Swap Space' (For Cloud VM only)
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

### Compile kernel (Choose either option)

#### Option 1: In terminal

Make sure you are in root mode.  

```bash
cd ~/linux-5.10.5/
make -j$(nproc)
```
It takes about 1~2 hrs to finish. Don't disconnect, don't close the terminal.  

#### Option 2: With `nohup` command (Recommended)
[Reference](https://www.runoob.com/linux/linux-comm-nohup.html)  

Make sure you are in root mode.  

```bash
cd ~/linux-5.10.5/
nohup make -j$(nproc)       # Does not accept any intput, run in backend process, only can be killed by killing pid
```
It takes about 1~2 hrs to finish.   

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

Modify `getname_kernel()` function:

```bash
vim ~/linux-5.10.5/fs/namei.c
```

then search for `struct filename * getname_kernel` (Hint: use search mode in vim as mentioned above, or line 212), add `extern` tag on it, then add `EXPORT_SYMBOL(getname_kernel);` behind (line 247), then save and quit.

Modify `kernel_clone()` function:

```bash
vim ~/linux-5.10.5/kernel/fork.c
```

then search for `pid_t kernel_clone(struct kernel_clone_args *args)` (Hint: use search mode in vim as mentioned above, or line 2416), add `extern` tag on it, then add `EXPORT_SYMBOL(kernel_clone);` behind (line 2495), then save and quit.

------

Then recompile & reinstall the kernel.  

Please start from the step:  

```
make -j$(nproc)
make modules_install
make install
```
Then reboot.  
To save your time, don't start from `make mrproper`. 



## Screenshot of the program output

### Program 1

Received SIGABRT signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig20.jpg" alt="fig20" style="zoom:50%;" />

------

Received SIGALRM signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig21.jpg" alt="fig21" style="zoom:50%;" />

------

Received SIGBUS signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig22.jpg" alt="fig22" style="zoom:50%;" />

------

Received SIGFPE signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig23.jpg" alt="fig23" style="zoom:50%;" />

------

Received SIGHUP signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig24.jpg" alt="fig24" style="zoom:50%;" />

------

Received SIGILL signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig25.jpg" alt="fig25" style="zoom:50%;" />

------

Received SIGINT signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig26.jpg" alt="fig26" style="zoom:50%;" />

------

Received SIGKILL signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig27.jpg" alt="fig27" style="zoom:50%;" />

------

Received SIGCHLD signal (normal termination):

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig28.jpg" alt="fig28" style="zoom:50%;" />

------

Received SIGPIPE signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig29.jpg" alt="fig29" style="zoom:50%;" />

------

Received SIGQUIT signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig30.jpg" alt="fig30" style="zoom:50%;" />

------

Received SIGSEGV signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig31.jpg" alt="fig31" style="zoom:50%;" />

------

Received SIGSTOP signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig32.jpg" alt="fig32" style="zoom:50%;" />

------

Received SIGCHLD signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig33.jpg" alt="fig33" style="zoom:50%;" />

------

Received SIGTRAP signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig34.jpg" alt="fig34" style="zoom:50%;" />




### Program 2

Received SIGBUS signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig4.jpg" alt="fig4" style="zoom:50%;" />

------

Received SIGHUP signal: 

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig6.jpg" alt="fig6" style="zoom:50%;" />


------

Received SIGINT signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig7.jpg" alt="fig7" style="zoom:50%;" />

------

Received SIGQUIT signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig8.jpg" alt="fig8" style="zoom:50%;" />

------

Received SIGILL signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig9.jpg" alt="fig9" style="zoom:50%;" />

------

Received SIGTRAP signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig10.jpg" alt="fig10" style="zoom:50%;" />

------

Received SIGABRT signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig11.jpg" alt="fig11" style="zoom:50%;" />

------

Received SIGFPE signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig12.jpg" alt="fig12" style="zoom:50%;" />

------

Received SIGKILL signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig13.jpg" alt="fig13" style="zoom:50%;" />

------

Received SIGSEGV signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig14.jpg" alt="fig14" style="zoom:50%;" />

------

Received SIGPIPE signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig15.jpg" alt="fig15" style="zoom:50%;" />

------

Received SIGALRM signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig16.jpg" alt="fig16" style="zoom:50%;" />

------

Received SIGTERM signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig17.jpg" alt="fig17" style="zoom:50%;" />

------

Received SIGSTOP signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig18.jpg" alt="fig18" style="zoom:50%;" />

------

Received SIGCHLD signal:

<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig19.jpg" alt="fig19" style="zoom:50%;" />



### Program 3



## Things I learned from the tasks

From this task, I learn how the program interact with kernel. Also, I learned how kernel works, and what we can do with the kernel. Most important is, I learn the methodology of how to deal with a large project. Like Linux kernel, it is implemented by thousands of deliciated functions and definitions. When we need to modify certain functions or to add some new features, we can first take a look at its source code. Also, computer science is a subject which everything is updating in a rapid way. Many useful materials can be found on Google, GitHub, etc. 

Also, I learned some useful technics of Linux command, which is very helpful for debugging. When something is stuck, we can use certain indicators (like `print` / `printf` / `printk` ) to locate the problem.

