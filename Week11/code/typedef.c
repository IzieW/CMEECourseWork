#include <stdio.h>
/* C allows you to make aliases for datatypes that might serve a particular function. 
typedef doesn't allow you to create your own datatypes, but can improve readability
in your code! */

int main(void)
{
    typedef char DNA_t; // DNA_t a type that can be used to make variables

    // Especially useful in conjunction with structs: 
    typedef struct { // Struct called DNA_sequence
        char *taxon_name;
        char *gene_name;
        int seq_length;
        DNA_t *sequence;
    } DNA_sequence;
}