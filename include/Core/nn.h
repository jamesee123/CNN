#include "CNN.h"

struct CNN_Structure 
{
    uint16_t hLayerCount;
    uint16_t inputCount;
    uint16_t* hiddenShape;
    enum CNN_ActivationFunctions* activationFunctions;
};

struct CNN_AI 
{
    uint16_t hLayerCount;
    uint16_t inputCount;
    uint16_t* hiddenShape;
    enum CNN_ActivationFunctions* activationFunctions;
    double*** weights;
    double** biases;
};

struct CNN_Structure CNN_CreateStructure(uint16_t hLayerCount, uint16_t inputCount, u_int16_t* hiddenShape, enum CNN_ActivationFunctions* activationFunctions);
void CNN_InitAI(struct CNN_AI* ai, struct CNN_Structure perams);
struct CNN_AI CNN_CreateAI(uint16_t hLayerCount, uint16_t inputCount, uint16_t* hiddenShape, enum CNN_ActivationFunctions* activationFunctions);
double* CNN_CalculateOutput(struct CNN_AI ai, double* input);