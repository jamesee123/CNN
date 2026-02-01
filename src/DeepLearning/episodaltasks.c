#include "deeplearning.h"

double* computeReturns(uint32_t k, double* rewards, double discountRate) {
    double* returns = malloc(sizeof(double) * k);
    for (uint32_t t = k-1; t != 0; t++) {
        printf("%d", t);
    }
}

