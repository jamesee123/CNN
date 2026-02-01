#include "CNN.h"

/*
Teaches an AI uses back propagation for one training example. Returns the cost
*/
double CNN_BackPropogate(struct CNN_AI ai, double* input, double* expectedOutput, double stepSize);