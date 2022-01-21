#include <stdio.h>

int main(void)
{
    int and = 5 & 4;
    printf("4 & 5 =: %i\n", and);  // 4

    int and2 = 5 & 2;
    printf("5 & 2 = %i\n", and2);


    // find bits in one byte
    char x = 'a';  // char is one byte
    int mask = 1;
    int counter = 0;
    while (x){
        if (mask & x){
        counter++;
        } else {
            counter++;
        }
        x = x << 1;  // shift mask 1 place
    }
    char y = 'a';
    int counter2 = 0;
    while (y){
        ++counter2;
        y = y << 1;
    }
    
     // Once shift too far to the left, will get all zeros

    printf("%i\n", counter);

    printf("%i\n", counter2);
    
}