#include <stdio.h>

int main(void){
    // Declaring strings similar to declaring an array

    char mystring1[] = "A really boring example of string";
    char mystring2[] = "Another boring string";

    // These can be written directly into printf

    printf(mystring1, mystring2, "\n");

    printf("\n%s\n%s\n", mystring1, mystring2);

    return 0;
}