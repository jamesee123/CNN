#include "CNN.h"

double* DEEP_InitValues(uint16_t k) {
    double* toReturn = malloc(sizeof(double) * k);
    for (uint16_t i = 0; i < k; i++)
    {
        toReturn[i] = 10;
    }
    return toReturn;
}

void DEEP_UpdateValues(double* expectedRewards, uint16_t n, double reward, double stepSize) {
    expectedRewards[n] = expectedRewards[n] + stepSize * (reward - expectedRewards[n]);
}

uint16_t DEEP_Greedy(double* estimatedValues, uint16_t k) {
    double maxValue = -INFINITY;
    uint16_t choice;
    for (uint16_t i = 0; i < k; i++) {
        if (estimatedValues[i] < maxValue) continue;
        choice = i;
        maxValue = estimatedValues[i];
    }

    return choice;
}

uint16_t DEEP_EGreedy(double* estimatedValues, uint16_t k, double epsilon) {
    bool goGreedy = ((double)rand() / (double)RAND_MAX) > epsilon;

    if (goGreedy) return DEEP_Greedy(estimatedValues, k);
    
    return rand() % k;
}

uint16_t DEEP_UpperConfidenceBound(double* estimatedValues, uint32_t* countPrevSelected, uint16_t k, uint32_t t, double degOfExploration) {
    double maxValue = -INFINITY;
    uint16_t choice;
    for (uint16_t i = 0; i < k; i++) {
        double val = estimatedValues[i] + degOfExploration * sqrt(log(t) / countPrevSelected[i]);
        if (val < maxValue) continue;
        choice = i;
        maxValue = val;
    }

    countPrevSelected[choice] ++;

    return choice;
}