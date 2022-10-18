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
int thread_ids[11] = {0,1,2,3,4,5,6,7,8,9,10};
int log_pos[9];
int stop_signal = 1;
int quit_signal = 0;


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
	while(stop_signal){
		pthread_mutex_lock(&count_mutex);
		/*  Check keyboard hits, to change frog's position or quit the game. */
		if(kbhit()){
			key_response = getchar();
			if(key_response == 'w' || key_response == 'W'){
				clean_original_path();
				frog.x -= 1;
				if(map[frog.x][frog.y] == ' ') pthread_cond_signal(&count_threshold_cv);
				map[frog.x][frog.y] = '0' ;
			}
			else if(key_response == 'a' || key_response == 'A'){
				clean_original_path();
				frog.y -= 1;
				if(map[frog.x][frog.y] == ' ' || frog.y == -1) pthread_cond_signal(&count_threshold_cv);
				map[frog.x][frog.y] = '0' ;
			}
			else if(key_response == 's' || key_response == 'S'){
				clean_original_path();
				frog.x += 1;
				if(map[frog.x][frog.y] == ' ' || frog.x == 11) pthread_cond_signal(&count_threshold_cv);
				map[frog.x][frog.y] = '0' ;
			}
			else if(key_response == 'd' || key_response == 'D'){
				clean_original_path();
				frog.y += 1;
				if(map[frog.x][frog.y] == ' '|| frog.y == 49) pthread_cond_signal(&count_threshold_cv);
				map[frog.x][frog.y] = '0' ;
			}
			else if(key_response == 'q' || key_response == 'Q'){
				quit_signal = 1;
				pthread_cond_signal(&count_threshold_cv);
			}
		}
		/*  Check game's status  */
		if(*log_no % 2 == 1){
			map[*log_no+1][log_pos[*log_no]] = ' ';
			map[*log_no+1][(log_pos[*log_no] + 15) % 49] = '=';
			log_pos[*log_no] = (log_pos[*log_no] + 1) % 49;
			if(frog.x == *log_no + 1){
				map[frog.x][frog.y] = '=';
				map[*log_no+1][(log_pos[*log_no]+48) % 49] = ' ';
				frog.y += 1;
				if(frog.y == 49) pthread_cond_signal(&count_threshold_cv);
				map[frog.x][frog.y] = '0';
			}
		}
		else{
			map[*log_no+1][(log_pos[*log_no] + 48) % 49] = '=';
			map[*log_no+1][(log_pos[*log_no] + 14) % 49] = ' ';
			log_pos[*log_no] = (log_pos[*log_no] + 48) % 49;
			if(frog.x == *log_no + 1){
				map[frog.x][frog.y] = '=';
				map[*log_no+1][(log_pos[*log_no] + 15) % 49] = ' ';
				frog.y -= 1;
				if(frog.y == -1) pthread_cond_signal(&count_threshold_cv);
				map[frog.x][frog.y] = '0';
			}
		}
		if(frog.x == 0){
			pthread_cond_signal(&count_threshold_cv);
		}
		pthread_mutex_unlock(&count_mutex);
		// 300000us
		usleep(100000);
	}
	pthread_exit(NULL);
}

void *print_pic( void *t ){
	int i, k;
	while(stop_signal){
		pthread_mutex_lock(&count_mutex);
		// system("clear");
		printf("\033[2J\n");
		for(i = 0; i <= ROW; ++i)	
			puts( map[i] );
		pthread_mutex_unlock(&count_mutex);
		usleep(100000);
	}
	pthread_exit(NULL);
}

void *check_status( void *t ){
	pthread_mutex_lock(&count_mutex);
	while(stop_signal){
		pthread_cond_wait(&count_threshold_cv, &count_mutex);
		stop_signal = 0;
	}
	pthread_mutex_unlock(&count_mutex);
	pthread_exit(NULL);
}

int main( int argc, char *argv[] ){

	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j , random_num; 

	// Configure command line output.
	printf("\033[?25l");
	// printf("\033[36m");	// Font color
	// printf("\033[43m");	// Background color

	// Initialize the position of the logs.
	for( i = 0; i < 9; i++){
		log_pos[i] = rand() % 49;
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
	pthread_t threads[11];
	pthread_attr_t attr;

	// Initialize mutex and conditional variable objects.
	pthread_mutex_init(&count_mutex, NULL);
	pthread_cond_init(&count_threshold_cv, NULL);
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	for( i = 0; i < 9; i++) {
		pthread_create(&threads[i],&attr, logs_move, (void*)&thread_ids[i]);
	}
	pthread_create(&threads[9],&attr, print_pic, (void*)&thread_ids[9]);
	pthread_create(&threads[10],&attr, check_status, (void*)&thread_ids[10]);
	for( i = 0; i < 9; i++) {
		pthread_join(threads[i], NULL);
	}
	pthread_join(threads[9], NULL);
	pthread_join(threads[10], NULL);
	printf("Main function ends.\n");

	// Clean up and exit
	pthread_attr_destroy(&attr);
	pthread_mutex_destroy(&count_mutex);
	pthread_cond_destroy(&count_threshold_cv);
	/*  Display the output for user: win, lose or quit.  */
	printf("\033[0m");	// Clean all color configuration
	printf("\033[2J");	// Clean the screen
	if(frog.x == 0){
		puts("You win");
	}
	else if(quit_signal){
		puts("You quit the game.");
	}
	else{
		puts("You lose the game.");
	}
	// Main thread exits
	pthread_exit(NULL);

	return 0;

}
