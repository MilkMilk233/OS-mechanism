#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50 


struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; 


char map[ROW+10][COLUMN] ; 
pthread_mutex_t count_mutex;
pthread_cond_t count_threshold_cv;
int thread_ids[9] = {0,1,2,3,4,5,6,7,8};
int log_pos[9];


// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}

void clean_original_path(void){
	if(frog.x != 0 && frog.x != 10){
		map[frog.x][frog.y] = '=';
	}
	else{
		map[frog.x][frog.y] = '|';
	}
}

void *logs_move( void *t ){
	/*  Move the logs  */
	int i = 0;
	int *log_no = (int*)t;
	char key_response;
	for(int k = 0; k < 10; k++){
		pthread_mutex_lock(&count_mutex);
		/*  Check keyboard hits, to change frog's position or quit the game. */
		if(kbhit()){
			key_response = getchar();
			if(key_response == 'w' || key_response == 'W'){
				clean_original_path();
				frog.x -= 1;
				map[frog.x][frog.y] = '0' ;
			}
			else if(key_response == 'a' || key_response == 'A'){
				clean_original_path();
				frog.y -= 1;
				map[frog.x][frog.y] = '0' ;
			}
			else if(key_response == 's' || key_response == 'S'){
				clean_original_path();
				frog.x += 1;
				map[frog.x][frog.y] = '0' ;
			}
			else if(key_response == 'd' || key_response == 'D'){
				clean_original_path();
				frog.y += 1;
				map[frog.x][frog.y] = '0' ;
			}
		}
		/*  Check game's status  */
		system("clear");
		if(*log_no % 2 == 0){
			map[*log_no+1][log_pos[*log_no]] = ' ';
			map[*log_no+1][(log_pos[*log_no] + 15) % 49] = '=';
			log_pos[*log_no] = (log_pos[*log_no] + 1) % 49;
		}
		else{
			map[*log_no+1][(log_pos[*log_no] - 1) % 49] = '=';
			map[*log_no+1][(log_pos[*log_no] + 14) % 49] = ' ';
			log_pos[*log_no] = (log_pos[*log_no] - 1) % 49;
		}
		/*  Print the map on the screen  */
		for( i = 0; i <= ROW; ++i)	
			puts( map[i] );
		pthread_mutex_unlock(&count_mutex);
		sleep(1);
	}
	pthread_exit(NULL);
}

void *check_status( void *t ){

}

int main( int argc, char *argv[] ){

	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j , random_num; 

	// Initialize the position of the logs.
	for( i = 0; i < 9; i++){
		log_pos[i] = rand() % 49;
		printf("%d\n",log_pos[i]);
	}
	
	// Draw logs initially.
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
		for( j = log_pos[i-1] ; j < log_pos[i-1] + 15; ++j)
			// printf("i = %d, j = %d, log_pos[i-1]=%d\n",i,j,log_pos[i-1]);
			map[i][j%49] = '=' ;
	}	

	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ;

	// for( j = 0; j < COLUMN - 1; ++j )	
	// 	map[0][j] = map[0][j] = '|' ;

	frog = Node( ROW, (COLUMN-1) / 2 ) ; 
	map[frog.x][frog.y] = '0' ; 

	//Print the map into screen
	for( i = 0; i <= ROW; ++i)	
		puts( map[i] );


	/*  Create pthreads for wood move and frog control.  */
	pthread_t threads[9];
	pthread_attr_t attr;

	// Initialize mutex and conditional variable objects.
	pthread_mutex_init(&count_mutex, NULL);
	pthread_cond_init(&count_threshold_cv, NULL);
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	for( i = 0; i < 9; i++) {
		pthread_create(&threads[i],&attr, logs_move, (void*)&thread_ids[i]);
	}
	for( i = 0; i < 9; i++) {
		pthread_join(threads[i], NULL);
	}
	printf("Main function ends.\n");

	// Clean up and exit
	pthread_attr_destroy(&attr);
	pthread_mutex_destroy(&count_mutex);
	pthread_cond_destroy(&count_threshold_cv);
	pthread_exit(NULL);

	/*  Display the output for user: win, lose or quit.  */
	system("clear");
	puts("Game Over!");

	return 0;

}
