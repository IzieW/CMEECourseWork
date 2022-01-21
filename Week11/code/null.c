#include <stdio.h>

int main(void){
    // C has no bounds checking.. so it cannot tell you a string's length

    int i = 0;
    char mystring[] = "A string printed character-by-character\n";

    // Can check length using a while loop
    while(mystring[i]) {
        printf("%c", mystring[i]);
        i++;
    }
    // Strings do have a hidden \0 (null) character which tells the compiler when the string ends
     char mystring2[] = "stringy!";
    // is the equivilent of
     mystring2[] = {'s', 't', 'r', 'i', 'n', 'g', 'y', '!', '\0'};

    // Strings are just arrays of characters with a terminating null character
    // Can use subscripting as well

     i = 0; 
    mystring2[] = "stringy!";

    printf("Character %i in mystring is: %c\n", i, mystring[i]);

    return 0;

}