#include "headers.h"

struct CNN_Gradient {
    double** activations;
    double** weightedSums;
    double*** cost2Weights;
    double**  cost2Biases;
    double**  cost2Activations;
    double** activations2WeightedSum;
    double cost;
};

void CNN_ComputeGradientLayer(double* input, struct CNN_AI ai, uint16_t l, struct CNN_Gradient gradient) {
    uint16_t layerSize = ai.hiddenShape[l];
    uint16_t prevLayerSize = l==0?ai.inputCount:ai.hiddenShape[l-1];
    double* previousActivations = l==0?input:gradient.activations[l-1];

    gradient.cost2Biases[l] = malloc(sizeof(double) * layerSize);
    gradient.cost2Weights[l] = malloc(sizeof(double*) * layerSize);
    
    for (uint16_t n = 0; n < layerSize; n++) {
        gradient.cost2Biases[l][n] = gradient.cost2Activations[l][n] * gradient.activations2WeightedSum[l][n];
        gradient.cost2Weights[l][n] = malloc(sizeof(double) * prevLayerSize);
        for (uint16_t m = 0; m < prevLayerSize; m++) {
            gradient.cost2Weights[l][n][m] = gradient.cost2Activations[l][n] * gradient.activations2WeightedSum[l][n] * previousActivations[m];
        }
    }

    if (l == 0) return;

    gradient.cost2Activations[l-1] = malloc(sizeof(double) * prevLayerSize);
    for (uint16_t m = 0; m < prevLayerSize; m++) {
        double sum = 0;
        for (uint16_t n = 0; n < layerSize; n++) {
            sum += ai.weights[l][n][m] * gradient.activations2WeightedSum[l][n] * gradient.cost2Activations[l][n];
        }
        gradient.cost2Activations[l-1][m] = sum;
    }
    CNN_ComputeGradientLayer(input, ai, l-1, gradient);
}

struct CNN_Gradient CNN_ComputeGradient(struct CNN_AI ai, double* input, double* expectedOutput) 
{
    struct CNN_Gradient gradient = {};
    gradient.activations = malloc(sizeof(double*) * ai.hLayerCount);
    gradient.weightedSums = malloc(sizeof(double*) * ai.hLayerCount);
    gradient.activations2WeightedSum = malloc(sizeof(double*) * ai.hLayerCount);

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

        gradient.activations[l] = malloc(sizeof(double) * layerSize);
        gradient.activations2WeightedSum[l] = malloc(sizeof(double) * layerSize);
        gradient.weightedSums[l] = malloc(sizeof(double) * layerSize);

        dotProduct(previousLayer, ai.weights[l], layerSize, prevLayerSize);
        sumArrays(previousLayer, ai.biases[l], layerSize);
        memcpy(gradient.weightedSums[l], previousLayer, sizeof(double)*layerSize);
        memcpy(gradient.activations2WeightedSum[l], previousLayer, sizeof(double)*layerSize);
        CNN_ApplyAFDerivative(gradient.activations2WeightedSum[l], layerSize, ai.activationFunctions[l]);
        CNN_ApplyAF(previousLayer, layerSize, ai.activationFunctions[l]);
        memcpy(gradient.activations[l], previousLayer, sizeof(double)*layerSize);
        prevLayerSize = layerSize;
    }

    gradient.cost = 0;
    double* dCostdActivations = malloc(sizeof(double) * ai.hiddenShape[ai.hLayerCount-1]);

    for (uint16_t o = 0; o < ai.hiddenShape[ai.hLayerCount-1]; o++) 
    {
        dCostdActivations[o] = 2 * (gradient.activations[ai.hLayerCount-1] - expectedOutput);
        gradient.cost += pow(gradient.activations[ai.hLayerCount-1][o] - expectedOutput[o], 2);
    }

    gradient.cost2Activations = malloc(sizeof(double*) * ai.hLayerCount);
    gradient.cost2Biases = malloc(sizeof(double*) * ai.hLayerCount);
    gradient.cost2Weights = malloc(sizeof(double**) * ai.hLayerCount);

    gradient.cost2Activations[ai.hLayerCount-1] = malloc(sizeof(double) * ai.hiddenShape[ai.hLayerCount-1]);
    for (uint16_t n = 0; n < ai.hiddenShape[ai.hLayerCount-1]; n++) {
        gradient.cost2Activations[ai.hLayerCount-1][n] = 2 * (previousLayer[n]-expectedOutput[n]); //TODO: MAKE THIS WORK AND THEN MAKE BACK_PROPOGATE_LAYER USE COST2ACTIVATIONS
    }

    CNN_ComputeGradientLayer(input, ai, ai.hLayerCount - 1, gradient);

    return gradient;
}

double CNN_BackPropogate(struct CNN_AI ai, double* input, double* expectedOutput, double stepSize)
{
    struct CNN_Gradient gradient = CNN_ComputeGradient(ai, input, expectedOutput);

    for (uint16_t l = 0; l < ai.hLayerCount; l++) {
        //printArray(gradient.activations2WeightedSum[l], ai.hiddenShape[l]);
        for (uint16_t n = 0; n < ai.hiddenShape[l]; n++) {
            ai.biases[l][n] += gradient.cost2Biases[l][n] * -stepSize;
            uint16_t prevLayerSize = l==0?ai.inputCount:ai.hiddenShape[l];
            for (uint16_t m = 0; m < prevLayerSize; m++) {
                //printf("%f\n", gradient.cost2Weights[l][n][m]);
                ai.weights[l][n][m] += gradient.cost2Weights[l][n][m] * -stepSize;
            }
        }
    }
    return gradient.cost;
}