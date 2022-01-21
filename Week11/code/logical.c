#include <stdio.h>
#include <stdbool.h>

int main(void)
{
    /* logical operators*/
    bool x = false; // def: 0
    bool y = true;  // def: 1
    
    if (x) {
        printf("x is true\n");
    }

    int i = 0;
    if (i == 0) {
        printf("i is true\n");
    } else {
        printf("i is true\n")
    }

    if (!i){
        printf("i is not true\n")
    }

    // binary logical operators:
    // && = and 
    // || = OR

    return 0;
}