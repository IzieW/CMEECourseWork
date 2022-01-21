#include <stdio.h>
/*Since C lacks bounds, working with arrays can be very dangerous. Worthwhile to 
create a C library that allows you to customise arrays and work safely with them!*/

/* Struct initialised below, now create a pair of special functions to create
and destroy int array*/
intarray_t *create_intarray(int maxsize) // return pointer to struct
{
    intarray_t *newarray_ptr = NULL; //pointer to struct
    newarray_ptr = (intarray_t*)calloc(sizeof(intarray)); // Allocates memory for intarray and
    // passes address back to newarray_ptr

}
int main(void)
{
    // WRAPPING ARRYAS IN A STRUCT
    typedef struct {
        int maxvals; // Max size an array can have
        int numvals; // Number of values you want in an array
        int head; // Keep track of last value when sequentially usign elments in array
        int *entries; // pointer to ints can be dynamically allocated according to size needs
    } intarray_t;

}