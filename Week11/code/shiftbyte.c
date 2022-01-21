#include <stdio.h>
#include <stdlib.h>

void printbitz(int x){
    // print bits in an integer
    int c = 0;
    unsigned mask = 1;
    while (mask) {
        if (mask & x) {
            printf("1");
        } else {
            printf("0");
        }
        ++c;
        if (c == 8){
            printf(" ");
            c = 0;
        }
        mask = mask << 1;
    }
    printf("\n");
}

int main(void)
{
    int x; 
    x = 88172666;  // fairly big numbers

    char* p;  // points to a character 
    // "A dirty trick"

    p = (char*)&x;  // "programmatically naughty": give it the address of an integer

    // (char*) typecast operation to silence the warning: we know that it is the case. 

    printf("Now p is: %i\n", *p);

    p[2];  // P an array of characters, four bytes long (since pointed to an integer, which is four bytes)
    // Can shift it 
    
    printbitz(p);

    p[2] = p[2] << 3;  // shift 3 to the left

    printbitz(p);

    return 0;
}