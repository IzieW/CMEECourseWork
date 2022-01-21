#include <stdio.h>

int main(void)
{
    // Write a programme that concatenates the following strings: 
    char string1[] = "The quick brown fox";
    char string2[] = "jumped over the lazy dog";

    char string3[sizeof(string1)+sizeof(string2)];

    int x = 1;
    int b = 0;
    int i; 
    int size = sizeof(string3);
    for (i=0; i<size; i++){
        if (i < sizeof(string2)){
        string3[i] = string1[i];
    } else {
        string3[i] = string2[b];
        ++b;
    }
    }

    for (i=0; i<size; i++){
        if (i == size){
            printf(" ");
        }
        printf("%c", string3[i]);
    }

    printf("\n");

}