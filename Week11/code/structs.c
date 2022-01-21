#include <stdio.h>
#include <string.h>

// Create an alias for script: 
typedef char DNA_base_t;

// Can then initialise ^^ characters throughout script just using DNA_base_t
// Can makes things easier to read + follow

// Structs aggregate types of data
// Structs and enums don't create any new memory, they instruct compiler
// how to interpret certain data types
// Central to object-orientated programming
// Class = general abstract idea
// Object = instance of that class
// Ex: class car, objects would be a car in the car park
// Ex: class phylogenetic tree, a specific data structure that reps a pylogenetic tree

// Want to create the definition for a type of thing

struct site_data {
    // define elements of it in terms of other data types the compiler knows about
    // Ex: struct stores a geographics points
    float latitude;
    float longitude;
    float elevation; 
    // Might include details of certain animals found there... 
    int observed_spp[500];  // set certain number of species
    int condition;  // some kind of scale of condition, degredation etc. 
};

// This struct does not take up any memory, but to create an instance of this
// it would: 

typedef struct site_data site_data_s; // Remove need below to spell out "struct site_data"
// can now just use "site_data_s.variable" to initialise various values
// Can even do the same name to make more concise: 

typedef struct site_data site_data; 


int main(void)
{
    // Create an object of class site_data
    struct site_data site1;
    struct site_data site2;
    struct site_data site3;

    struct site_data mysites[3];  // makes three sites with all of the variables above allocated to memory

    // Let's initialise a struct: 
    // Can individually set each one... quite tedious 
    // Set values in site1: use member select operator "."
    site1.latitude = 74.3444;

    // Instead can use a function that lives in string.h (header above)
    // void *Memset(void*str, int c, sizet_t)
    // Function takes a pointer to void, integer and size, and returns a pointer to void
    // In data at ptr, Will set every bite to the value int c for sizet_t many bites
    // Ex: set every bite in site 1 to zero

    memset(&site1, 0, sizeof(struct site_data));  // Point to site1, change every value to zero for as many values are in site1
    memset(&site2, 0, sizeof(struct site_data));
    memset(&site3, 0, sizeof(struct site_data));

    // What about for array mysites?
    // No need for pointer, as will treat an array as a pointer

    memset(mysites, 0, 3 * sizeof(struct site_data)); // Set to three times the size of our struct, since an array of three
    // or can just see the size of that array
    memset(mysites, 0, sizeof(mysites));
    printf("The latitude of site 1 is: %f\n", site1.latitude);
    printf("The longitude of site 1 is : %f\n", site1.longitude);  // Give me garbage
    
    return 0;
}