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
		//Child process
		if(pid != 0){
			printf("I'm the Parent Process, my pid = %d\n",getpid());
			wait(&status);
			printf("Parent process receives SIGCHLD signal\n");
			if(status == 0){
				printf("Normal termination with EXIT STATUS = 0\n");
				exit(0);
			}
			else{
				// Caught some failure
				exit(1);
			}
			
		}
		else{
			int i;
			char *arg[argc];
			printf("I'm the Child Process, my pid = %d\n",getpid());
			for(i = 0; i < argc-1; i++){
				arg[i] = argv[i+1];
			}
			arg[argc-1] = NULL;
			printf("Child process start to execute test program:");
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
