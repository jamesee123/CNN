#include "neat.h"

struct NEAT_Generation NEAT_CreateGeneration(struct CNN_Structure structure, u_int32_t populationSize)
{
    struct NEAT_Generation toReturn = {structure, malloc(sizeof(struct CNN_AI) * populationSize), populationSize};
    NEAT_InitPopulation(toReturn);

    return toReturn;
}

struct CNN_AI NEAT_MutateAI(struct CNN_AI parent, double mutationRate) {
    struct CNN_AI toReturn;
    toReturn.activationFunctions = parent.activationFunctions;
    toReturn.biases = malloc(sizeof(double) * parent.hLayerCount);
    toReturn.weights = malloc(sizeof(double**) * parent.hLayerCount);
    toReturn.hiddenShape = parent.hiddenShape;
    toReturn.inputCount = parent.inputCount;
    toReturn.hLayerCount = parent.hLayerCount;

    uint16_t prevLayerSize = parent.inputCount;
    for (uint16_t l = 0; l < parent.hLayerCount; l++) {
        uint16_t layerSize = parent.hiddenShape[l];
        toReturn.weights[l] = malloc(sizeof(double*) * layerSize);

        toReturn.biases[l] = mutateArray(layerSize, parent.biases[l], mutationRate);

        for (u_int16_t n = 0; n < layerSize; n++) {
            toReturn.weights[l][n] = mutateArray(prevLayerSize, parent.weights[l][n], mutationRate);
        }
        prevLayerSize = layerSize;
    }

    return toReturn;
}

void NEAT_InitPopulation(struct NEAT_Generation gen) 
{
    for (uint32_t c = 0; c < gen.populationSize; c++) 
    {
        CNN_InitAI(gen.population+c, gen.structure);
    }
}

void NEAT_SimulationStep(struct NEAT_Generation gen, double* fitnesses, double mutationRate)
{
    struct CNN_AI* oldGen = malloc(sizeof(struct CNN_AI) * gen.populationSize);
    memcpy(oldGen, gen.population, sizeof(struct CNN_AI) * gen.populationSize);

    bool* selected = (bool*)malloc(sizeof(bool) * gen.populationSize);

    double weightSum = 0;

    for (uint32_t i = 0; i < gen.populationSize; i++) {
        selected[i] = false;
        weightSum += fitnesses[i];
    }

    for (uint32_t i = 0; i < gen.populationSize/2; i++) {
        double randomValue = (double)rand() / (double)RAND_MAX * weightSum;
        uint32_t j = 0;
        for (j = 0; j < gen.populationSize; j++) {
            if (selected[j]) continue;
            randomValue -= fitnesses[j];
            if (randomValue <= 0) {
                weightSum -= fitnesses[i];
                selected[j] = true;
                break;
            };
        }

        gen.population[i * 2] = oldGen[j];
        gen.population[i * 2 + 1] = NEAT_MutateAI(oldGen[j], mutationRate);
    }
}
