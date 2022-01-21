#include <stdio.h>

int main(void)
{
    // bitwise operatiosn make it extremely easy to do fast calculations on
    // categorical data variables (ex: the presence or abscence of values in certain
    // sites or phenotypic traits)

    // Consider binary presence/abscence data across 8 species at two sites
    // (each array corresponds to a cite)

    int site1[] = {0, 0, 1, 1, 0, 1, 1, 1};
    int site2[] = {1, 1, 0, 1, 0, 0, 1, 1};

    // to determine which species are present in both sites, could use
    // a laborious for loop:
    for (i=0; i < 8; ++i){
        if (site[i] == 1 && site2[i] == 1){
            return 1;
        }
    }
    // or... can do a simple bitwise operations
    if (site1 & site2){
        return 1
    }

    // Ancestral states
    int d1 = sp1 & sp2;
    int d2 = d1 & sp3;

    // Fitch parsimoney
    // Can represent GTCA easily using integers
    int A_ = 1;
    int C_ = 2;
    int G_ = 4;
    int T_ = 8;
    
    // However, there are far more bits in an integer than we need, so can use characters instead
    char A_ = (char)1;
    char C_ = (char)(1 << 1)
    char G_ = (char)(1 << 2)
    char T_ = (char)(1 << 3)

    // Can write a subroutine that converts a DNA base into a single set bit:
    char ret = 0; 
    if (base == 'A' || base == 'a') {
        ret = A_;
    }
    else if (base == 'C' || base == 'c') {
        ret = C_;
    }
    else if (base == 'G' || base == 'G') {
        ret = G_;
    }
    else if (base == 'T' || base == 'T') {
        ret = T_;
    }
    else if (base == 'Y' || base == 'y') {
        ret = C_ | T_;
    }
    else if (base == 'R' || base == 'r') {
        ret = A_ | G_;
    }


}