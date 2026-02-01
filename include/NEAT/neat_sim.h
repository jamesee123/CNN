#include "CNN.h"

/*
An object that represents a population of neural networks
*/
struct NEAT_Generation 
{
    struct CNN_Structure structure;
    struct CNN_AI* population;
    uint32_t populationSize;
};

/*
Creates a population of neural networks
*/
struct NEAT_Generation NEAT_CreateGeneration(struct CNN_Structure structure, u_int32_t populationSize);

/*
Initializes each neural network in a population of neural networks
*/
void NEAT_InitPopulation(struct NEAT_Generation gen);

/*
Runs natural selection on a population of neural networks
*/
void NEAT_SimulationStep(struct NEAT_Generation gen, double* fitnesses, double mutationRate);

/*
Slightly adjusts the perameters of a neural network
*/
struct CNN_AI NEAT_MutateAI(struct CNN_AI parent, double mutationRate);