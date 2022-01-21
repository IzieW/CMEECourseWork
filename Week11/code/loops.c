#include <stdio.h>

int main(void)
{
    // Three different styles of loop in c

    int x = 0;
    
    // While loop
    while (x < 5) {
        printf("one loop\n");
        x++; // x = x + 1; x += 1
    }
    // Do-while...
    x = 0;
    do {
        ++x;
    } while (x < 10);
    
    // For..
    for (x = 0; x < 10; ++x) {
        // First set initiates, second set is condition, third set is transformaton
        printf("Lets see how this works...\n");
    }
    // Goto:


}