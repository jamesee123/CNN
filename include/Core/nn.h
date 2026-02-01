#include "CNN.h"

/*
A struct that can be used to represent the shape of a NN, and the activations each layer has
*/
struct CNN_Structure 
{
    uint16_t hLayerCount;
    uint16_t inputCount;
    uint16_t* hiddenShape;
    enum CNN_ActivationFunctions* activationFunctions;
};

/*
A structure that contains the perameters behind a NN.
*/
struct CNN_AI 
{
    uint16_t hLayerCount;
    uint16_t inputCount;
    uint16_t* hiddenShape;
    enum CNN_ActivationFunctions* activationFunctions;
    double*** weights;
    double** biases;
};

/*
A function that creates a CNN_Structure object given the perameters necessary
*/
struct CNN_Structure CNN_CreateStructure(uint16_t hLayerCount, uint16_t inputCount, u_int16_t* hiddenShape, enum CNN_ActivationFunctions* activationFunctions);

/*
A function that sets the perameters for a CNN_AI object given a CNN_Structure object
*/
void CNN_InitAI(struct CNN_AI* ai, struct CNN_Structure perams);

/*
A function that creates a CNN_AI object given the perameters necessary
*/
struct CNN_AI CNN_CreateAI(uint16_t hLayerCount, uint16_t inputCount, uint16_t* hiddenShape, enum CNN_ActivationFunctions* activationFunctions);

/*
A function that propagates an input through a CNN_AI object, returning an output
*/
double* CNN_CalculateOutput(struct CNN_AI ai, double* input);