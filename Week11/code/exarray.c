#include <stdio.h>

int main(void)
{
int array1[] = {1, 2, 3, 4, 5, 6};
int array2[] = {7, 8, 9};
int array3[10];

//  Concatenate array1 and array2 in array3
int i;
int b = 0;

for (i = 0; i < 10; ++i){
    if (i < 6){
    array3[i] = array1[i];
    } else {
    array3[i] = array2[b];
    ++b;
    }
}

for (i = 0; i < 10; i ++){
    printf("%i", array3[i]);
}

printf("\n");

return 0;
}
