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

/* Search the /proc file with the help of `opendir()`, `readdir()` and
  `closedir()` Trying to get information about process & thread */
void search_process_info() {
  int pid = 0, sub_pid = 0;
  struct dirent *dir_file, *subdir_file;
  char *folder_name;
  DIR *dir, subdir;  // Store the structure of the folder

  if (!(dir = opendir("/proc"))) {
    printf("Can't open '/proc': Permission denied.\n");
    return -1;
  }
  while ((dir_file = readdir(dir)) != NULL) {
    /*  Find the hidden thread folder, e.g. ".243" */
    if ((pid = atoi(dir_file->d_name)) == 0) {
      continue;
    } else {  // store in pidinfo (name and pid and ppid)
      /* First look for threads */
      if (!(subdir = opendir("/proc/%s/task\n", dir_file->d_name))) {
        printf("Can't open '/proc/%s/task': Permission denied.\n",
               dir_file->d_name);
      } else {
        while ((subdir_file == readdir(subdir)) != NULL) {
          if ((sub_pid = atoi(dir_file->d_name)) == 0) {
            continue;
          } else {
            pidinfos[pid_count].pid = pid;
            pidinfos[pid_count].ppid =
                readprocessname_ppid(pid, pidinfos[pid_count].name);
            pidinfos[pid_count].tpid = sub_pid;
            assert(pidinfos[pid_count].ppid > -1);
            // printf("%d (%s)
            // %d\n",pidinfos[pid_count].pid,pidinfos[pid_count].name,pidinfos[pid_count].ppid);
            pid_count++;
          }
        }
      }
      closedir(subdir);
      /* Then look for pids*/
      pidinfos[pid_count].pid = pid;
      pidinfos[pid_count].ppid =
          readprocessname_ppid(pid, pidinfos[pid_count].name);
      assert(pidinfos[pid_count].ppid > -1);
      // printf("%d (%s)
      // %d\n",pidinfos[pid_count].pid,pidinfos[pid_count].name,pidinfos[pid_count].ppid);
      pid_count++;
    }
  }
  closedir(dir);
}

int main(int argc, char *argv[]) {
  if (argc == 1) {
    printf("This is the default case.\n");
    search_process_info();
  } else {
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
