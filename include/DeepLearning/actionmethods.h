#include "deeplearning.h"

enum ActionMethod {
    Greedy,
    Epsilon_Greedy,
    Upper_Confidence_Bound
};

double* DEEP_InitValues(uint16_t k);
void DEEP_UpdateValues(double* expectedRewards, uint16_t n, double reward, double stepSize);
uint16_t DEEP_Greedy(double* expectedRewards, uint16_t k);
uint16_t DEEP_EGreedy(double* expectedRewards, uint16_t k, double epsilon);
uint16_t DEEP_UpperConfidenceBound(double* expectedRewards, uint32_t* countPrevSelected, uint16_t k, uint32_t t, double degOfExploration);