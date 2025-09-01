#include "headers.h"

void CNN_InitPerameters(struct CNN_AI* toInit) 
{
    toInit->biases = (double**)malloc(sizeof(double*) * toInit->hLayerCount);
    toInit->weights = (double***)malloc(sizeof(double**) * toInit->hLayerCount);

    uint16_t previousLayerSize = toInit->inputCount;
    for (uint16_t l = 0; l < toInit->hLayerCount; l++) 
    {
        uint16_t currentLayerSize = toInit->hiddenShape[l];
        toInit->weights[l] = (double**)malloc(sizeof(double*) * currentLayerSize);
        toInit->biases[l] = (double*)malloc(sizeof(double) * currentLayerSize);
        randomArray(toInit->biases[l], currentLayerSize);

        for (uint16_t n = 0; n < currentLayerSize; n++) 
        {
            toInit->weights[l][n] = (double*)malloc(sizeof(double) * previousLayerSize);
            randomArray(toInit->weights[l][n], previousLayerSize);
        }

        previousLayerSize = currentLayerSize;
    }
};

struct CNN_Structure CNN_CreateStructure(uint16_t hLayerCount, uint16_t inputCount, u_int16_t* hiddenShape, enum CNN_ActivationFunctions* activationFunctions)
{
    return (struct CNN_Structure) {hLayerCount, inputCount, hiddenShape, activationFunctions};
}

void CNN_InitAI(struct CNN_AI* ai, struct CNN_Structure perams) 
{
    ai->activationFunctions = perams.activationFunctions;
    ai->hiddenShape = perams.hiddenShape;
    ai->hLayerCount = perams.hLayerCount;
    ai->inputCount = perams.inputCount;
    CNN_InitPerameters(ai);
}

struct CNN_AI CNN_CreateAI(uint16_t hLayerCount, uint16_t inputCount, uint16_t* hiddenShape, enum CNN_ActivationFunctions* activationFunctions) 
{
    struct CNN_AI toReturn;// = {hLayerCount, inputCount};
    toReturn.hLayerCount = hLayerCount;
    toReturn.inputCount = inputCount;
    toReturn.activationFunctions = malloc(sizeof(enum CNN_ActivationFunctions) * hLayerCount);
    memcpy(toReturn.activationFunctions, activationFunctions, sizeof(enum CNN_ActivationFunctions) * hLayerCount);
    toReturn.hiddenShape = (uint16_t*)malloc(sizeof(uint16_t) * hLayerCount);
    memcpy(toReturn.hiddenShape, hiddenShape, sizeof(uint16_t) * hLayerCount);
    CNN_InitPerameters(&toReturn);

    return toReturn;
}

double* CNN_CalculateOutput(struct CNN_AI ai, double* input) 
{
    uint16_t outputMemory = ai.inputCount;

    for (uint16_t i = 0; i < ai.hLayerCount; i++) 
    {
        outputMemory = ai.hiddenShape[i]>outputMemory? ai.hiddenShape[i] : outputMemory;
    }

    double* previousLayer = malloc(sizeof(double) * outputMemory);
    memcpy(previousLayer, input, ai.inputCount * sizeof(double));

    uint16_t prevLayerSize = ai.inputCount;

    for (uint16_t l = 0; l < ai.hLayerCount; l++) 
    {
        uint16_t layerSize = ai.hiddenShape[l];
        dotProduct(previousLayer, ai.weights[l], layerSize, prevLayerSize);
        sumArrays(previousLayer, ai.biases[l], layerSize);
        CNN_ApplyAF(previousLayer, layerSize, ai.activationFunctions[l]);
        prevLayerSize = layerSize;
    }

    return previousLayer;
}