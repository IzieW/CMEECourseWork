#include <stdio.h>
// How to use structs in C
int main(void)
{
    struct site_data{ // Initiate struct site_data: a general class of structure 
        float lat;
        float longit; // "long" already a key word, so best to avoid
        float elev; 
    }

    // Declaring an instance of a structure
    struct site_data mysite1;
    // Accessing/writing to a struct done using the . operator
    mysite1.lat = 32.045;
    mysite1.longit = -104.181;

    // Same operator used to access data inside the struct:
    int current_lat = mysite1.lat;
    int current_long = mysite1.longit;

    // Ex: Using pointers in structs... similar to using pointers anywehre
    struct int_ptrs {
        int *pt1;
        int *pt2;
    }

    // Assign values to pt1 just like any other variables: 
    int my_int = 0; 

    struct int_ptrs twoints;
    twoints.pt1 = &my_int;

    // Now can assign values to my_int through pt1
    *twoints.pt1 = 12;
    // The . operand has higher precedence, thus above statement equivilent to:
    // "Select member of pt1 in twoints, then dereference it"

    // Pointers to structures: 
    // Especially powerful and useful

    struct site_data *site_data_ptr;
    site_data_ptr = &mysite1; // points to my site 1
    (*mysite1).lat = 24.118;

    // operation is common enough that it has been abbreviated as so: 
    mysite -> lat;
}