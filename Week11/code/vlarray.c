#include <stdio.h>
#include <stdlib.h>

int* create_int_array(int nelems)
{
    // Returns an array
    int* newarray;

    newarray = malloc(sizeof(int) * nelems); // Declaration of malloc:void* malloc(size_t n);
    //newarray = calloc(nelems, sizeof(int)); // safer
    return newarray;
}

int main(int argc, char *argv[])
{
    // Where argv[] is an array of strings from the command line
    // arvc is count of arguments
    int i; 
    unsigned int nelems;

    // Expect one argument from user, and it should be an integer greater than 0
    if (argc != 2) {
        printf("ERROR: program requires 1 (and only 1) integer argument!\n");
        return 1;
    }

    nelems = atoi(argv[1]); // atoi turns string into integer
    if (nelems == 0) {
        printf("ERROR: input must be non-zero\n");
        return 1;
    }


    char *userinputnumber = argv[1];  

    for (i = 0; i < nelems; ++i) {
        printf("%i ", myintegers[i]);
    }
    printf("\n");

    return 0;
}