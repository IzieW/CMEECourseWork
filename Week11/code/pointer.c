#include <stdio.h>

int main(void){
    int i = 0;
    int *int_ptr = NULL;

    printf("The value of i BEFORE indirection: %i\n", i);

    int_ptr = &i;
     *int_ptr = 4; 

    printf("The value of i AFTER indirection: %i\n", i);

    // Pointers and arrays
    int site_populations[150] = {0};
    int *populations_ptr = NULL;

    populations_ptr = site_populations; // New pointer automatically set to the first element in the array

    // You can declar an array of pointers - helpful with large memory objects
    char *site_names[] = {"parking lot", "Cricket lawn", "Manor house",
    "Silwood bottom", "The reactor", "Japanese garden",};

    i = 0; 
    for (i = 0; i < 6; ++i) {
        printf("%s\n", site_names[i]);
    }

    // Can also do pointer arithmatic
    // Useful for when an pointer addresses an element of an array or block of memory

    populations_ptr = populations_ptr + 4;
    *populations_ptr; // Dereferences the fifth element of site_populations 
    // can also do: 
    *(populations_ptr + 4) = 0; // Set the value at site_populations[4]

    // Can use incremenet operators on pointers
    for (i = 0; i < 4; ++i){
        ++populations_ptr;
    }

    printf("Pointer is now at: %ls", populations_ptr);

}