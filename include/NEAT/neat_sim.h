#include "CNN.h"

struct NEAT_Generation 
{
    struct CNN_Structure structure;
    struct CNN_AI* population;
    uint32_t populationSize;
};

struct NEAT_Generation NEAT_CreateGeneration(struct CNN_Structure structure, u_int32_t populationSize);
void NEAT_InitPopulation(struct NEAT_Generation gen);
void NEAT_SimulationStep(struct NEAT_Generation gen, double* fitnesses, double mutationRate);
struct CNN_AI NEAT_MutateAI(struct CNN_AI parent, double mutationRate);