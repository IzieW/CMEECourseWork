#include <stdio.h> // Preprocessor directive- forward declaration that tells C compiler 
// about the function printf (different functionalities to include)
// Essentially copies script and adds in any textual substitutions from package

/* Multiline
 comments
ended with star-slash
*/

int main (void) // Main function, equivilent to python. Int- returns integer, takes void arguments
{
    /* A comment */
    printf("Well, hello...\n");  // All statements in C end with a semicolon

    return 0;  // Everything went OK. Return 0 to the OS
}

// Run with gcc or g++ to compile, then run the executable with ./
// White space above not needed, but much more readable :)) 