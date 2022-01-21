#include <stdio.h>

int main(void){
    // Impliment the Fitch algorithm

    int sp1 = 1;
    int sp2 = 2;
    int sp3 = 3;
    int d1; 
    int d2;

    // First pass
    if (sp1 & sp2){
        d1 = (sp1 & sp2);
    } else {
        d1 = (sp1 | sp2);
    }

    if ((sp1 & sp3) && (sp1 & sp3)){
        printf("hella yas\n");
    }
}