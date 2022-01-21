#include <stdio.h>

void index_through_array(int numbers[5], int index)
{
    while (index < 5){
        printf("Elements %i: %i\n", index, numbers[index]);
        ++index;
    }

    printf("The value of index at end of function call: %i\n", index);
    return;
}

int main(void){
    int index = 0;
    int mynums[] = {19, 81, 4, 8, 10};

    printf("Value of i before function call: %i\n", index);
    index_through_array(mynums, *index);
    printf("Value of i after function call: %i\n", index);

    return 0;
}