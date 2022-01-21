#include <stdio.h>

int main(void)
{
    // Write a programme that determines the width in bits of a single byte on your system
    // Assume Char is one byte 

    char byte;
    int count = 0;

    while (byte){
        count++;
    }

    printf("There are %i bits in a byte on my system\n", count); 

    return 0;
}