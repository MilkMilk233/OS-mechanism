#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[]){

	/* fork a child process */
	int status;
	printf("Process start to folk\n");
	pid_t pid = fork();
	/* execute test program */ 
	if(pid < 0){
		printf("Fork error!\n");
	}
	else{
		//Parent process
		if(pid != 0){
			printf("I'm the Parent Process, my pid = %d\n",getpid());
			waitpid(pid, &status, WUNTRACED);
			printf("Parent process receives SIGCHLD signal\n");
			if(WIFEXITED(status)){
				printf("Normal termination with EXIT STATUS = %d\n",WEXITSTATUS(status));
			}
			else if(WIFSIGNALED(status)){
				if(WTERMSIG(status) == 1){
					printf("CHILD EXECUTION FAILED: Receive SIGHUP, the process is hang up.\n");
				}
				else if(WTERMSIG(status) == 2){
					printf("CHILD EXECUTION FAILED: Receive SIGINT, the process is interrupted.\n");
				}
				else if(WTERMSIG(status) == 3){
					printf("CHILD EXECUTION FAILED: Receive SIGQUIT, the process is quited.\n");
				}
				else if(WTERMSIG(status) == 4){
					printf("CHILD EXECUTION FAILED: Receive SIGILL, the process gets illegal instruction.\n");
				}
				else if(WTERMSIG(status) == 5){
					printf("CHILD EXECUTION FAILED: Receive SIGTRAP, the process is terminated by trap signal.\n");
				}
				else if(WTERMSIG(status) == 6){
					printf("CHILD EXECUTION FAILED: Receive SIGABRT, the process is abort.\n");
				}
				else if(WTERMSIG(status) == 7){
					printf("CHILD EXECUTION FAILED: Receive SIGBUS, the process gets bus error.\n");
				}
				else if(WTERMSIG(status) == 8){
					printf("CHILD EXECUTION FAILED: Receive SIGFPE, the process gets floating point exception.\n");
				}
				else if(WTERMSIG(status) == 9){
					printf("CHILD EXECUTION FAILED: Receive SIGKILL, the process is killed.\n");
				}
				else if(WTERMSIG(status) == 11){
					printf("CHILD EXECUTION FAILED: Receive SIGSEGV, the process uses invalid memory reference.\n");
				}
				else if(WTERMSIG(status) == 13){
					printf("CHILD EXECUTION FAILED: Receive SIGPIPE, the process writes to pipe with no readers.\n");
				}
				else if(WTERMSIG(status) == 14){
					printf("CHILD EXECUTION FAILED: Receive SIGALRM, the process is terminated by alarm signal.\n");
				}
				else if(WTERMSIG(status) == 15){
					printf("CHILD EXECUTION FAILED: Receive SIGTERM, the process is terminated by termaniation signal.\n");
				}
				else{
					printf("CHILD EXECUTION FAILED: %d\n", WTERMSIG(status));
				}
				
			}
			else if(WIFSTOPPED(status)){
				if(WSTOPSIG(status) == 19){
					printf("CHILD PROCESS STOPPED: Receive SIGSTOP signal\n");
				}
				else{
					printf("CHILD PROCESS STOPPED: %d\n", WSTOPSIG(status));
				}
			}
			else{
				printf("CHILD PROCESS CONTINUED\n");
			}
			exit(0);
			
		}
		// Child process
		else{
			int i;
			char *arg[argc];
			printf("I'm the Child Process, my pid = %d\n",getpid());
			for(i = 0; i < argc-1; i++){
				arg[i] = argv[i+1];
			}
			arg[argc-1] = NULL;
			printf("Child process start to execute test process:");
			execve(arg[0],arg,NULL);

			printf("Continue to run original child process!\n");
			perror("execve");
			exit(EXIT_FAILURE);
		}
	}
	return 0;
	
	/* wait for child process terminates */
	
	/* check child process'  termination status */
	
	
}
