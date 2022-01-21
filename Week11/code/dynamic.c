#include <stdio.h>

/* Often you will not know the size needs of an array before runtime. 
The Malloc and Calloc functions (available in the stdio library) make calls to 
the operating system for a desired amount of memory.

If the memory is available, these functions will return the address (a pointer!) to
the newly reserved space! 

If memory is unavailable, they will return a NULL, but this is rarely the case.*/

// Their prototypes: 
void *malloc(size_t size); // (returns an address to type void)
void *calloc(size_t num_items, size_t size_of_elements);

/* size_t is a new data type, usually an unsigned long integer
Express the size of data type in terms of system bytes- can be obtained using sizeof() */


// Malloc and Calloc return a pointer to an area of memory or any size we want
// (As long as it's available on our system, that is)
int main(void){
    // if you need memory for 20 integers, using malloc; 
    int *_20_ints = NULL;
    _20_ints = (int*)malloc(20 * sizeof(int)); // Type cast (int*) to tell compiler what we want

    // using calloc: 
    _20_ints = (int*)calloc(20, sizeof(int)); // Calloc stands for "clear allocation"
    
    /* Calloc will clear memory by setting all bits to 0, meanwhile malloc won't 
     do anything to initialise the memory.
     
     Calloc therefore is safer to use, though a bit slower. Malloc is the "base function"
     meanwhile Calloc wraps malloc in extra code tha clears the allocated memory */

     // Using dynamic memory allocation: 
     // Create a nucleotide sequence of arbitrary length

     int numsites = 0;
     char *nucleotide_sites = NULL; // Pointer to nothing

     nucelotide_sites = (char*)calloc(num_sites, sizeof(char));

     // Can check if calloc succeeded quite quickly by adding safety rails
     if(!nucleotide_sites = (char*)calloc(num_sites, sizeof(char))){
         printf("Error: unable to allocate sufficient memeory\n");
         exit(EXIT_FAILURE)
     }
}
