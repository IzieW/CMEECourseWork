#include <stdio.h>

int main(void){
    int result;
    int myarray[] = {1, 2, 3, 4, 5};

    myarray[0] = myarray[1] + myarray[2];

    printf("The result is: %i\n", myarray[0]);

    // We don't always know what size an array will be, so can use:
    int n_entries = 10;
    // ... later in the programme
    int myarray2[n_entries];
    // Most compilers will support variable sized arrays.. but practice therefore not entirely portable
    int x;

    for(x=0; x < 12; x++){
        myarray2[x] = x;
        printf("%i\n", myarray2[x]);  // Print numbers 1-10
    }    

    return 0;
}