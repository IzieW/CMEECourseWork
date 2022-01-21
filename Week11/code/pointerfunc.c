#include <stdio.h>

int my_function_that_uses_a_pointer(int *param_ptr){
    return 0;
}

int *function_that_returns_a_pointer(int * oddsandevents){
    return 0;
}

int *first_odd_num(int* oddsandevens, int arraymax){
    int i = 0;
    while (!(*oddsandevens % 2) && i < arraymax){
        ++oddsandevens;
        ++i;
    }
    return oddsandevens;
}

int main(void){
    // For function that takes a pointer
    // simply call by passing in the name of a pointer variable: 
    
    int i = 0;
    int *int_ptr = &i;
    
    my_function_that_uses_a_pointer(int_ptr);

    int arraymax = 5;
    int intarray[] = {2, 4, 6, 7, 5};
    int *result = NULL;

    result = first_odd_num(intarray, arraymax);
    printf("First odd: %i\n", *result);
}