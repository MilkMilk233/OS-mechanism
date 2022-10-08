#include <assert.h>
#include <dirent.h>  // Open the folder in /proc
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

typedef struct pidinfo {
  char name[50];
  __pid_t pid;
  __pid_t ppid;
  __pid_t tid;
} PidInfo;
/* Define a struct, containing its pid, ppid, tid. */
PidInfo pidinfos[10000];
int pid_count = 0;

void read_ppid_and_name(__pid_t pid, __pid_t sub_pid) {
  char *str_pid = (char *)malloc(sizeof(char) * 20);
  sprintf(str_pid, "%d", pid);
  char *str_sub_pid = (char *)malloc(sizeof(char) * 20);
  sprintf(str_sub_pid, "%d", sub_pid);
  char path[30] = "/proc/";
  strcat(path, str_pid);
  strcat(path, "/task/");
  strcat(path, str_sub_pid);
  strcat(path, "/stat");
  FILE *fp = fopen(path, "r");
  if (fp) {
    char name[50];
    char i;              // Process Status, seize a seat only
    __pid_t _pid, ppid;  // seize a seat only
    fscanf(fp, "%d (%s %c %d", &_pid, name, &i, &ppid);
    name[strlen(name) - 1] = '\0';
    strcpy(pidinfos[pid_count].name, name);
    pidinfos[pid_count].ppid = ppid;
    printf("name=%s,ppid=%d\n", name, ppid);
    fclose(fp);
    free(str_pid);
    free(str_sub_pid);
    return;
  } else {
    printf("Error: file open failed at '%s'\n", path);
    exit(1);
  }
}

/* Search the /proc file with the help of `opendir()`, `readdir()` and
  `closedir()` Trying to get information about process & thread */
void search_process_info() {
  int pid = 0, sub_pid = 0;
  struct dirent *dir_file, *subdir_file;
  char *folder_name;
  DIR *dir, *subdir;  // Store the structure of the folder

  if (!(dir = opendir("/proc"))) {
    printf("Can't open '/proc': Permission denied.\n");
    return;
  }
  while ((dir_file = readdir(dir)) != NULL) {
    /*  Find the hidden thread folder, e.g. ".243" */
    if ((pid = atoi(dir_file->d_name)) == 0) {
      continue;
    } else {  // store in pidinfo (name and pid and ppid)
      /* First look for threads */
      char path[30] = "/proc/";
      strcat(path, dir_file->d_name);
      strcat(path, "/task");
      if (!(subdir = opendir(path))) {
        printf("Can't open '%s': Permission denied.\n", path);
      } else {
        while ((subdir_file = readdir(subdir)) != NULL) {
          if ((sub_pid = atoi(subdir_file->d_name)) == 0) {
            continue;
          } else {
            pidinfos[pid_count].pid = pid;
            pidinfos[pid_count].tid = sub_pid;
            printf("Read: pid=%d,tpid=%d,", pid, sub_pid);
            read_ppid_and_name(pid, sub_pid);
            pid_count++;
          }
        }
      }
      closedir(subdir);
    }
  }
  closedir(dir);
  return;
}

int main(int argc, char *argv[]) {
  if (argc == 1) {
    printf("This is the default case.\n");
    search_process_info();
  } else {
    int o;
    const char *optstring = "Vclap";
    while ((o = getopt(argc, argv, optstring)) != -1) {
      switch (o) {
        case 'V':
          printf(
              "pstree (PSmisc) 22.21 \nCopyright (C) 1993-2009 Werner "
              "Almesberger and Craig Small\nPSmisc comes with ABSOLUTELY NO "
              "WARRANTY.\nThis is free software, and you are welcome to "
              "redistribute it under\nthe terms of the GNU General Public "
              "License.\nFor more information about these matters, see the "
              "files "
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
          printf("Usage: pstree [ -a ] [ -c ] [ -l ] [ -p ]\n");
          printf("       pstree -V\n");
          printf("Display a tree of processes.\n\n");
          printf(
              "       pstree -a                  show command line "
              "arguments\n");
          printf(
              "       pstree -c                  don't compact identical "
              "subtrees\n");
          printf(
              "       pstree -l                  don't truncate long lines\n");
          printf("       pstree -p                  show PIDs; implies -c\n");
          printf(
              "       pstree -V                  display version "
              "information\n");
          break;
      }
    }
  }
  return 0;
}
