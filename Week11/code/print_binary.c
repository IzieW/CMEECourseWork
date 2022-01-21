#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]){  // Always the two main arguments that come through
int mask = 1;

char* cast = argv[1];  // Fix to first argument given
int arg = atoi(cast);

int i = 0;
int bin[32]; // Saves in reverse: 
    while (mask){
        if (mask & arg){
            bin[i] = 1;
        } else {
            bin[i] = 0;
        }
    i++;
    mask = mask << 1;
    }

// Flips back around: 
for (i=31; i >= 0; i--){
    printf("%i", bin[i]);
}

    printf("\n");

    //printf("Number given is: %i\n", *argv[1]);

 
        printf("%i\n", arg);
    
    return 0;
}