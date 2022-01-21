#include <stdio.h>

int main(void)
{
    // Get the lowest set bit in a variable
    int var = 100;

    int mask = 1; 
    int min; 


    while (mask){
        if (var & mask){
            printf("The lowest set is %i\n", mask);
            break;
        } else {
            mask = mask << 1;
        }
    }
    
    int anothermask = 1; 

    while(anothermask){
        if (anothermask & var){
            printf("1");
        } else {
            printf("0");
        }
        anothermask = anothermask << 1;
    }
    printf("\n");

}