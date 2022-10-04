#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>

MODULE_LICENSE("GPL");

/* Define Struct & extern function */

struct wait_opts {
	enum pid_type wo_type;
	int	wo_flags;
	struct pid *wo_pid;

	struct waitid_info *wo_info;
	int wo_stat;
	struct rusage *wo_rusage;

	wait_queue_entry_t child_wait;
	int notask_error;
};

struct kernel_clone_args {
	u64 flags;
	int __user *pidfd;
	int __user *child_tid;
	int __user *parent_tid;
	int exit_signal;
	unsigned long stack;
	unsigned long stack_size;
	unsigned long tls;
	pid_t *set_tid;
	/* Number of elements in *set_tid */
	size_t set_tid_size;
	int cgroup;
	struct cgroup *cgrp;
	struct css_set *cset;
};

static struct task_struct *task;

extern long do_wait(struct wait_opts *wo);
extern int do_execve(struct filename *filename,
	const char __user *const __user *__argv,
	const char __user *const __user *__envp);
extern pid_t kernel_clone(struct kernel_clone_args *args);
extern struct filename *getname_kernel(const char * filename);

int my_exec(void){
	int result;
	const char path[] = "/home/vagrant/CSC3150/Assignment1/source/program2/test";
	const char *const argv[] = {path, NULL, NULL};
	const char *const envp[] = {"HOME=/", "PATH=/sbin:/user/sbin:/bin:/usr/bin", NULL};
	struct filename * file_path = getname_kernel(path);
	printk("[program2] : child process");
	result = do_execve(file_path, argv, envp);
	if(!result){
		return 0;
	}
	printk("[program2] : child process failed.");
	do_exit(result);
}

int my_wait(pid_t pid){
	int status;
	int a;
	
	// int terminatedStatus;
	struct wait_opts wo;
	struct pid * wo_pid = NULL;
	enum pid_type type;
	type = PIDTYPE_PID;
	wo_pid = find_get_pid(pid);

	wo.wo_type   = type;
	wo.wo_pid    = wo_pid;
	wo.wo_flags  = WEXITED|WUNTRACED;
	wo.wo_info   = NULL;
	wo.wo_stat   = (int __user*) &status;
	wo.wo_rusage = NULL;

	do_wait(&wo);
	a = wo.wo_stat;

	put_pid(wo_pid);

	return a;
}

//implement fork function
int my_fork(void *argc){
	//set default sigaction for current process
	int status, i;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for(i=0;i<_NSIG;i++){
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}
	
	pid_t pid;
	struct kernel_clone_args kargs;
	/* fork a process using kernel_clone or kernel_thread */
	kargs.flags = SIGCHLD;
	kargs.pidfd = NULL;
	kargs.child_tid= NULL;
	kargs.parent_tid = NULL;
	kargs.exit_signal = 0;
	kargs.stack = (unsigned long)& my_exec;
	unsigned long stack_size;
	unsigned long tls;
	pid_t *set_tid;
	/* Number of elements in *set_tid */
	size_t set_tid_size;
	int cgroup;
	struct cgroup *cgrp;
	struct css_set *cset;

	pid = kernel_clone(SIGCHLD,NULL,NULL,NULL,0,(unsigned long)& my_exec,0,0,NULL,0,0,NULL,NULL);
	/* execute a test program in child process */
	printk("[program2] : The Child process has pid = %d\n", pid);
	printk("[program2] : This is the parent process, pid = %d\n", (int)current->pid);
	/* wait until child process terminates */
	status = my_wait(pid);
	printk("[program2] : The return signal is %d", status);
	
	return 0;
}

static int __init program2_init(void){

	printk("[program2] : module_init Chen Zhixin 120090222\n");
	
	/* write your code here */
	printk("[program2] : module_init create kthread start");
	/* create a kernel thread to run my_fork */

	task = kthread_create(&my_fork, NULL, "MyThread");
	if(!IS_ERR(task)){
		printk("[program2] : module_init kthread start");
		wake_up_process(task);
	}
	
	return 0;
}

static void __exit program2_exit(void){
	printk("[program2] : Module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);
