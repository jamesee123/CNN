#include "deeplearning.h"

/*
An enum object that represents an action method
*/
enum ActionMethod {
    Greedy,
    Epsilon_Greedy,
    Upper_Confidence_Bound
};

/*
Returns random expected rewards for a policy function
*/
double* DEEP_InitValues(uint16_t k);

/*
Adjusts a policy function based on the actual reward
*/
void DEEP_UpdateValues(double* expectedRewards, uint16_t n, double reward, double stepSize);

/*
Runs the Greedy action function on an array of expected rewards
*/
uint16_t DEEP_Greedy(double* expectedRewards, uint16_t k);

/*
Runs the EGreedy action function on an array of expected rewards
*/
uint16_t DEEP_EGreedy(double* expectedRewards, uint16_t k, double epsilon);

/*
Runs the Upper Confidence Bound action function on an array of expected rewards
*/
uint16_t DEEP_UpperConfidenceBound(double* expectedRewards, uint32_t* countPrevSelected, uint16_t k, uint32_t t, double degOfExploration);