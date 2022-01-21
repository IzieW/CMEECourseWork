#include <stdio.h>

int main(void)
{
    signed char schar;  // signed isn't needed, but let's be explicit
    unsigned char uchar; 
    
    //schar = 1;  // in binary: 00000001
    uchar = 1;  // 00000001

    printf("%i\n", schar);
    printf("%u\n", uchar);

    printf("%i\n", schar << 8);
    printf("%u\n", uchar << 8);    

    int counter = 0;
    while(schar){
        counter++;
        schar = schar << 1;
    }

    printf("The count is %i", counter);

    return 0;
}