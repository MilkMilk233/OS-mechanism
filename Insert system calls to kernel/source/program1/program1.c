#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
	/* fork a child process */
	int status;
	printf("Process start to folk\n");
	pid_t pid = fork();
	/* execute test program */
	if (pid < 0) {
		printf("Fork error!\n");
	} else {
		// Parent process
		if (pid != 0) {
			printf("I'm the Parent Process, my pid = %d\n",
			       getpid());
			/* wait for child process terminates */
			waitpid(pid, &status, WUNTRACED);
			printf("Parent process receives SIGCHLD signal\n");
			/* check child process'  termination status */
			if (WIFEXITED(status)) {
				printf("Normal termination with EXIT STATUS = %d\n",
				       WEXITSTATUS(status));
			} else if (WIFSIGNALED(status)) {
				if (WTERMSIG(status) == 1) {
					printf("child process get SIGHUP signal\n");
				} else if (WTERMSIG(status) == 2) {
					printf("child process get SIGINT signal\n");
				} else if (WTERMSIG(status) == 3) {
					printf("child process get SIGQUIT signal\n");
				} else if (WTERMSIG(status) == 4) {
					printf("child process get SIGILL signal\n");
				} else if (WTERMSIG(status) == 5) {
					printf("child process get SIGTRAP signal\n");
				} else if (WTERMSIG(status) == 6) {
					printf("child process get SIGABRT signal\n");
				} else if (WTERMSIG(status) == 7) {
					printf("child process get SIGBUS signal\n");
				} else if (WTERMSIG(status) == 8) {
					printf("child process get SIGFPE signal\n");
				} else if (WTERMSIG(status) == 9) {
					printf("child process get SIGKILL signal\n");
				} else if (WTERMSIG(status) == 11) {
					printf("child process get SIGSEGV signal\n");
				} else if (WTERMSIG(status) == 13) {
					printf("child process get SIGPIPE signal\n");
				} else if (WTERMSIG(status) == 14) {
					printf("child process get SIGALRM signal\n");
				} else if (WTERMSIG(status) == 15) {
					printf("child process get SIGTERM signal\n");
				} else {
					printf("CHILD EXECUTION FAILED: %d\n",
					       WTERMSIG(status));
				}

			} else if (WIFSTOPPED(status)) {
				if (WSTOPSIG(status) == 19) {
					printf("child process get SIGSTOP signal\n");
				} else {
					printf("CHILD PROCESS STOPPED: %d\n",
					       WSTOPSIG(status));
				}
			} else {
				printf("CHILD PROCESS CONTINUED\n");
			}
			exit(0);

		}
		// Child process
		else {
			int i;
			char *arg[argc];
			printf("I'm the Child Process, my pid = %d\n",
			       getpid());
			for (i = 0; i < argc - 1; i++) {
				arg[i] = argv[i + 1];
			}
			arg[argc - 1] = NULL;
			printf("Child process start to execute test process:");
			execve(arg[0], arg, NULL);

			printf("Continue to run original child process!\n");
			perror("execve");
			exit(EXIT_FAILURE);
		}
	}
	return 0;
}
