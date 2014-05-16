#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char **argv)
{
    int rank, size;
    int check, i, j;
    int cpu_count, numcpus, cpunum;

    static int thread_id=0, num_threads=1;

    static cpu_set_t cpusetmask;
    static size_t cpusetsize=sizeof(cpu_set_t);

    int charcnt;
    static char cpus_string[400];
#pragma omp threadprivate(cpusetmask,cpusetsize,thread_id,num_threads,cpus_string)

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
#pragma omp parallel default(none) shared(rank,size) private(i,j,check)
    {
#ifdef _OPENMP
        thread_id=omp_get_thread_num();
        num_threads=omp_get_num_threads();
#endif
        check=sched_getaffinity(0,cpusetsize,&cpusetmask);
        if(check==-1){
            printf("Error in sched_getaffinity\n");
            MPI_Abort(MPI_COMM_WORLD,37);
        }
    }
    if(rank==0){
        fflush(stdout);
    }
    for(i=0;i<size;i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if(i==rank) {
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(rank,size,stdout) private(i,j,cpunum,check,cpu_count,numcpus,charcnt)
            for(j=0;j<num_threads;j++){
                if(j==thread_id){
#endif
                    cpu_count=CPU_COUNT(&cpusetmask);
                    charcnt=0;
                    charcnt=sprintf(cpus_string,"On pe %d and thread %d the number of cpus available is %d, ",
                            rank, thread_id, cpu_count);
                    if(cpu_count==1){
                        charcnt+=sprintf(&(cpus_string[charcnt]),"this being ");
                        for(cpunum=0;cpunum<CPU_SETSIZE;cpunum++){
                            if(CPU_ISSET(cpunum,&cpusetmask)){
                                charcnt+=sprintf(&(cpus_string[charcnt]),"%d\n",cpunum);
                                break;
                            }
                        }
                    }else{
                        charcnt+=sprintf(&(cpus_string[charcnt]),"these being ");
                        numcpus=0;
                        for(cpunum=0;cpunum<CPU_SETSIZE;cpunum++){
                            if(CPU_ISSET(cpunum,&cpusetmask)){
                                charcnt+=sprintf(&(cpus_string[charcnt]),"%d",cpunum);
                                numcpus++;
                                if(numcpus==cpu_count){
                                    charcnt+=sprintf(&(cpus_string[charcnt]),"\n");
                                    break;
                                }else{
                                    charcnt+=sprintf(&(cpus_string[charcnt]),", ");
                                }
                            }
                        }
                    }
                    printf("%s",cpus_string);
                    fflush(stdout);
#ifdef _OPENMP
                }
            }
#endif
        }
    }
    MPI_Finalize();
}
