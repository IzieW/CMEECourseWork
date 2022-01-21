#include <stdio.h>

int doubler(int i)
{
    // Take in an integer and double it
    i = i * 2;
    return i;
}

void array_doubler(int arr[], int nelems)
{
    // Languages like python know how large an array is, but C doesn't
    int i;
    for (i = 0; i < nelems; ++i){  // Essentially for i in arr:
        arr[i] = doubler(arr[i]);
    }
}

void print_intarray(int arr[], int nelems)
{
    int i;
    for (i = 0; i < nelems; ++i) {
        printf("%i", arr[i]);
    }
    printf("\n");
}

int main(void)
{
    int x = 7;
    int integs[5];

    int i;
    for (i = 0; i < 5; ++i) {
        integs[i] = i + 1;
    }

    x = doubler(x);  // Need to explicitly re-define x as doubler(x) to get value
    printf("The value of x: %i\n", x);
    //  When x is sent to doubler(x), a copy of x is made that is then acted on in the function
    // Thus need to redefine x as that copy

    array_doubler(integs, 5);  // However, with arrays, the above re-specification is not needed
    // Since C has no way of telling array length, it cannot copy an entire array over to a function
    // In the same way we saw with integers. 
    // Instead it passes a copy of a reference to the original array (point reference)

    // Distinction of values type vs. reference type data
    // here, x is a value type and integs is a reference type. It moves around in the programme by referencing the original
    // Reference like an address. 
    print_intarray(integs, 5);

    return 0;
}