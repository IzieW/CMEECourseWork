#include <stdio.h>

int main(void){
    char pal[] = "palindrome";

    int i;
    int b = 0;
    char reverse[sizeof(pal)];

    for (i=9; i>=0; i--){
        reverse[b] = pal[i];
        ++b;
    }

    printf("The reverse of %s is: %s\n", pal, reverse);
}