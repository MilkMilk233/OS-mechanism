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

1. `do_execve()`
2. `do_wait()`:
3. `kernel_clone()`:
4. `getname_kernel()`:
5. 

For function design, I separate into three functions:

- `my_fork()`:
- `my_wait()`:
- `my_exec()`:





<img src="https://ly-blog.oss-cn-shenzhen.aliyuncs.com/static/CSC3150/fig2.jpg" style="zoom:50%;" />

### Bonus

In bonus, I implemented a pstree function.



## How to set up development environment?

Reference



## Screenshot of the program output

### Program 1



### Program 2



### Program 3



## Things I learned from the tasks

From this task, I learn how the program interact with kernel. Also, I learned how kernel works, and what we can do with the kernel. Most important is, I learn the methodology of how to deal with a large project. Like Linux kernel, it is implemented by thousands of deliciated functions and definitions. When we need to modify certain functions or to add some new features, we can first take a look at its source code. Also, computer science is a subject which everything is updating in a rapid way. Many useful materials can be found on Google, GitHub, etc. 

Also, I learned some useful technics of Linux command, which is very helpful for debugging. When something is stuck, we can use certain indicators (like `print` / `printf` / `printk` ) to locate the problem.

