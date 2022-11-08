# CSC3150 Assignment 3 - Report
Chen Zhixin, 120090222

[TOC]

## Running Environment

**All my program run on <u>CSC4005's HPC Cluster</u>**

Kernel Version

```sh
[120090222@node21 Assignment3]$ uname -r
3.10.0-862.el7.x86_64
```

OS version

```sh
Centos 7.x
```

CUDA version

```sh
[120090222@node21 ~]$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Jun__8_16:49:14_PDT_2022
Cuda compilation tools, release 11.7, V11.7.99
Build cuda_11.7.r11.7/compiler.31442593_0
```

GPU/CPU Information (From intro session of CSC4005)

```
• Configuration
1 login node
20 Intel CPU cores (40 logic cores)
100GB RAM
1 Nvidia Quadro RTX 4000 GPU
• 30+ compute node
Each compute node has:
20 Intel CPU cores (40 logic cores)
100GB RAM
1 Nvidia Quadro RTX 4000 GPU
```



## Execution steps of running my program

File structure:

```sh
.
├── bonus
│   ├── data.bin
│   ├── main.cu
│   ├── slurm.sh
│   ├── user_program.cu
│   ├── virtual_memory.cu
│   └── virtual_memory.h
├── report.pdf
└── source
    ├── data.bin
    ├── main.cu
    ├── slurm.sh
    ├── user_program.cu
    ├── virtual_memory.cu
    └── virtual_memory.h
```

### Main

Assert you run on CSC4005 HPC cluster.

```sh
cd ./source/
nvcc --relocatable-device-code=true main.cu user_program.cu virtual_memory.cu -o test
bash slurm.sh
```

### Bonus

Assert you run on CSC4005 HPC cluster.

```sh
cd ./bonus/
nvcc --relocatable-device-code=true main.cu user_program.cu virtual_memory.cu -o test
bash slurm.sh
```



## Design Idea

### Main

#### Overview

CUDA is a general parallel computing platform and programming model based on NVIDIA CPUs. Programming based on CUDA can use the parallel computing engine of GPUs to solve more complex computing problems more efficiently. In this assignment, We use CUDA's **global memory** and **shared memory** to simulate **traditional CPU physical memory** and **secondary memory**, and finally implement the **paging mechanism**.

#### What is paging?

Traditional CPU physical memory has the characteristics of **high speed** and **easy access**. But it is **expensive** and has **little storage capacity**. In contrast, secondary storage has a slower speed, but is cheap and has huge capacity. Paging is such a technology that can magically expand the capacity of physical memory with secondary storage. This part of secondary storage expansion is usually called **virtual memory**. We divide data into many small chunks. When some chunks of data in the physical memory is not used for a long time, they will be transferred to the secondary memory, thus ensuring that the data stored in the physical memory is used by the CPU at a high frequency, and improving the overall operating efficiency of the CPU. This is the working principle of paging.

As mentioned above, we need to keep the frequently used data blocks in physical memory, while the infrequently used data blocks are stored in secondary memory. So how do we know which data stores should be placed? The answer is that we use a structure called **page table** to store information. When data is stored in the physical memory, the page table will also store the credentials of the data. We can judge whether the data is stored in the physical memory according to whether the required data exists in the page table. **If it does not exist, it is in secondary storage**. Therefore, the logic becomes clear: when we need a data block, we first send a request to the page table to see whether the data block is stored in physical memory. If we do not find the results in the page table, we need to spend more time to go deep into the secondary storage to find the data we need.

In addition, to simulate the real situation, I also implemented a **secondary storage page table**. When the program needs to enter the secondary storage to obtain data, it will search from the secondary storage table and convert the corresponding virtual address to the location where the data is stored in the secondary storage.

#### Swap strategy

We will use the **LRU (Least recent used) algorithm** to maintain the page table and ensure that data is used frequently by the CPU. To put it simply, when we cannot find a target in the physical memory and need to access the secondary memory, we first select a data block that is **least frequently used** from the current page table and **put it back into the secondary memory**; Then **exchange** a data block that we need in the secondary memory **back** **to the original location**.

![illustrator1](https://video.milkmilk.cloud/static/CSC3150/illustrator1.jpg)

#### PTE, From Virtual Address(VA) to Physical Memory

In addition to recording whether data blocks exist in physical memory, page table has another function: it is **a converter from virtual address to physical address**.

Before continuing, it is necessary to explain several words:

- Page: Just the "data block" we mentioned above.
- Page table entry(PTE): The unit that makes up the page table contains VA-PA pair, valid bit, LRU bit and other key information (to be mentioned later).
- Physical address: Location of data block in physical memory.
- Virtual address: Location of data block in program data. 
- Page number: The position of the page table entry(PTE).

Each program has a unique virtual address, but the memory used to store data is out of order. Therefore, we need to establish a one-to-one relationship between the virtual address and the physical address through the page table, so that we can easily obtain the physical address for storing data through the virtual address. Here we can think of the page table as a "translator", and the one-to-one relationship is stored in page table entry (PTE).

The following table tells us the size limit of each part of this task:

![swap](https://video.milkmilk.cloud/static/CSC3150/size.jpg)

**From this table, we can clearly see our task goal: we need to store up to 160KB of input into memory (of course, it is unrealistic to store all 128KB in physical memory, because it is only 32KB in size, so the excess part needs to be stored in secondary storage), and then extract 128KB of data from memory into output.**

Based on the flow chart and existing information, we can know:

- The physical memory is only 32KB in size and can only hold 1/5 of the input data. The remaining 4/5 are destined to be placed in the secondary memory.
- We split the data by 32 bytes. Each small piece of data is the smallest unit of exchange, which we call "page". For 128KB input, it can be divided into 4096 pages, and 1024 of them will have the opportunity to be placed in physical storage.
- For the 1024 pages stored on the physical storage, we will create a page table containing 1024 PTEs to manage their information. And for the 4096 pages stored on the disk storage, we will create a page table containing 4096 PTEs to manage their information.
- Since the total memory allocated to the Page table is 16KB, the memory allocated to each PTE is (16KB/1024) = 16 bytes = 4* unsigned integer(32 bit).

Let's talk about how I designed PTE to achieve all the set goals. I only use 1/2 of the specified limit, that is, each PTE uses two unsigned integers (2 * 32bit) to record the necessary information.

First unsigned integer: [32bits] -> [1*Valid bit(invalid = 1) | 11\* Unused bit | 4\* Thread Number | 16\* Virtual Address ]

First unsigned integer: [32bits] -> [LRU Ranking]

*(Since 160KB input/output can be divided into 5120 pages (i.e. 2 ^ 12 + 2^10), we only need 13 digits to represent the number of each page (i.e. Virtual address))*

#### Workflow: Read / Write / Snapshot

![illustrator2](https://video.milkmilk.cloud/static/CSC3150/illustrator2.jpg)

------

![illustrator3](https://video.milkmilk.cloud/static/CSC3150/illustrator3.jpg)

------



### Bonus

I implemented bonus part on the basis of **Version 2**:

>  The four threads perform the same and whole process of testcase1 four  times in total. This means that the addresses of the four threads are the same during  read and write operations. (The same task is done four times by four threads) In this  version, the next process overwrites the content of the previous process.



## Page fault number, also how does it come out

When we need to request a data that is not in the physical (shared) memory, a page fault will occur.

**As mentioned above**, at this time, we need to go deep into the secondary storage to find the pages we need and exchange them with the least frequently used pages stored in physical memory.

For example, when we write input between [0, 1023*32], the PTEs are not full in physical memory. But when we write another (1023\*32+1)th input, the page fault will occur, and the 1st page(least used one) in the physical memory will be put back to the secondary memory, while the 1st page in the secondary memory will be put to the position of the 1st page in the physical memory.

------

For main part, 

- Testcase #1 will produce 8193 page faults.
- Testcase #2 will produce 9215 page faults.

For bonus part (Version 2), 

- Testcase #1 will produce 8193 * 4 = 32772 page faults.

## Screenshot of the program output

### Main

Testcase #1, up to 128KB occupancy in total

![illustrator2](https://video.milkmilk.cloud/static/CSC3150/testcase1.jpg)

Testcase #2, up to 160KB occupancy in total

![illustrator2](https://video.milkmilk.cloud/static/CSC3150/testcase2.jpg)

### Bonus

Testcase #1, up to 128KB*4 occupancy in total

![illustrator2](https://video.milkmilk.cloud/static/CSC3150/testcase3.jpg)



## Problems I met in this assignment, and the solutions

- It took me a long time to figure out what page is, and then I thought for a long time about how to apply the page mechanism to this program.
- At first, I didn't know the memory limit of the secondary memory table so I posted questions on the forum, and received positive and quick response.
- I used one-to-one mapping between VA and storage, and made the first draft of 128KB version. However, with the further clarification of the task, the total memory expanded to 160KB. I had to override the rewrite and refactor more than 70% of the code.

- I once tried to complete the physical memory page table with only 4 KB, that is, 1/4 of the memory limit. But considering the possibility that the LRU index can grow indefinitely, I gave up the aggressive approach, chose a more conservative implementation method, which uses 8 KB, that is, 1/4 of the memory limit to complete the job.



## What did I learn from this assignment?

- I have a slight glance at CUDA programming.
- I learned some important feature about CUDA programming, like the block, threads' arrangement, memory sharing, variable and function type etc.
- I learned the mechanism of the paging\paging table\inverted paging table.
- I gain experience in handling programs with complex states.
- Debug skills going up. 
- More practice. 
- Project experience.