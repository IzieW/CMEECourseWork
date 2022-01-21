#include <stdio.h>

int main(void){
    // Return the square of myarray[i]

    int i = 0;
    int myarray[] = {1, 2, 3, 4, 5};

    for(i = 0; i < 5; i++){
        myarray[i] = myarray[i] * myarray[i];
    printf("The value at index %i in my array: %i\n", i, myarray[i]);
    }
}