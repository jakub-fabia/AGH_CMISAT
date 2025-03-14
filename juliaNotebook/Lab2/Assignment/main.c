#include <stdio.h>
#include <gsl/gsl_ieee_utils.h>

int main(){
    float x = 3e-35;
    for(int i = 0; i <= 34; i++){
        x = x / (2.0);
        gsl_ieee_printf_float(&x);
        printf("\n");
    }
    return 0;
}