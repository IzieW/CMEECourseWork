#include <stdio.h>

int main(void)
{
    int a;
    int b;
    int c;

    a = 1 + 2;
    b = 7;
    c = a + b;

    printf("The result of a + b using c: %i\n", c);
    printf("The result of a + b: %i\n", a + b + 3);

    float d;
    float e;
    
    // e = b;
    //d = e / a;

    d = (float)b / a;  //Type cast operator to specify variable type

    printf("%f\n", d);

    return 0;
}