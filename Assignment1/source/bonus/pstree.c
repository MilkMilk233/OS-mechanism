#include <assert.h>
#include <dirent.h>   // Open the folder in /proc
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>  

typedef struct pidinfo {
  char name[50];
  __pid_t pid;
  __pid_t ppid;
  __pid_t tid;
} PidInfo;
/* Define a struct, containing its pid, ppid, tid. */
PidInfo pidinfos[10000];

int main(int argc, char *argv[]) {
  if(argc == 1){
    printf("This is the default case.\n");
  }
  else{
    int o;
    const char *optstring =
        "Vc:lap";  // 有三个选项-abc，其中c选项后有冒号，所以后面必须有参数
    while ((o = getopt(argc, argv, optstring)) != -1) {
      switch (o) {
        case 'V':
          printf(
              "pstree (PSmisc) 22.21 \nCopyright (C) 1993-2009 Werner "
              "Almesberger and Craig Small\nPSmisc comes with ABSOLUTELY NO "
              "WARRANTY.\nThis is free software, and you are welcome to "
              "redistribute it under\nthe terms of the GNU General Public "
              "License.\nFor more information about these matters, see the files "
              "named COPYING.\n");
          break;
        case 'c':
          printf("opt is c, oprarg is: %s\n", optarg);
          break;
        case 'l':
          printf("opt is l, oprarg is: %s\n", optarg);
          break;
        case 'a':
          printf("opt is a, oprarg is: %s\n", optarg);
          break;
        case 'p':
          printf("opt is p, oprarg is: %s\n", optarg);
          break;
        case '?':
          printf("Usage: pstree [ -c ]\n");
          printf("       pstree -V\n");
          printf("Display a tree of processes.\n\n");
          printf("       pstree -V                  display version information\n");
          break;
      }
    }
  }
  return 0;
}
