#include <stdio.h>

enum my_error_t {
    // Declare an enumerated type
    MYPROG_SUCCESS, // evaluates to 0
    UNEXPECTED_NULLPTR,  // 1
    OUT_OF_BOUNDS,  // 2

    MY_ERROR_MAX  // 3
};

int main(void)
{
    enum my_error_t err;  // Define own error type

    const int arraymax = 5;
    int values[arraymax];
    int userval = 5;

    if (userval < arraymax) {
        printf("Value %i is: %i\n", userval, values[userval]);
    } else {
        err = OUT_OF_BOUNDS;
    }

    return err;
}