#include <stdio.h>

int main(void)
{
    // One bit is a binary digit: 0 or 1
    // 1 byte == 8 bits wide (though not true of all systems)
    // Each integer is 1 byte or 8 bits. 
    // Because of their binary:
    // int 0 = {00000000};
    // int 1 = {00000001};
    // int 2 = {00000010};

    // bits are "set" from right to left, with the most significant bit on the right
    // and least significant on the left. 
    
    // A signed int it one that can be negative, so has a lower positive range of integers
    // Unsigned ints cannot be negative, thus have a higher range of positive values
    // In signed ints, -1 is 11111111 whereas in unsigned ints 11111111 is 255:
    // which is the largest unsigned integer possible. 
    // The largest 8-bit signed integer possible would be 01111111, 127 (half of 254)


    // On most modern computers, int is 32-bits (4 bytes)
    // long int is 64-bit (8 bytes)
    // Short is 16 bits (2 bytes)

    // One character is one byte, 8 bits
            // Hence why in the earlier examples, 256 went to zero 
            // x was marked as a character, thus maxxed out capacity at 255
    
    // Bitwise operators:
        // | OR
        // & AND
        // ^ exlusive OR
        // ~ bitwise inverse (ones compliment)
        // >> right-shift operator
        // left-shift operator

    // First three work my aligning two variables "by the bit" and comparing them

    // ex: 
    // 00000100 4
    // 00000101 5
    // -------- &
    // 00000100 4
    // Both 4 and 5 share the 3rd set bit, so the result is only the 3rd set bit (4)

    // written in c: 
    int result = 5 & 4;
    printf("%i\n", result);


    // Ex: 
    // 01100010 98
    // 01110110 118
    // -------- &
    // 01100010 98

    int next_result = 98 & 118;
    printf("%i\n", next_result);

    // Ex: 
    // 01100010 98
    // 01110110 118
    // -------- |
    // 01110110 118

    int example = 98 | 118;
    printf("%i\n", example);

    // Ex: 
    // 01100010 98
    // 01010110 
    // -------- ^ 
    // 00110100

    // Ex:
    // 01010110
    // -------- ~ (flips the bits into inverse)
    // 10101001

    // Ex: 
    // Shift operators shift bits either left or right by x positions
    // 6 << 1 - shifts all bits to the left by one 
    // 6 00000110
    // 6 << 1: 00001100

    int shifted = 6 << 1;
    printf("%i", shifted);
    return 0;
}