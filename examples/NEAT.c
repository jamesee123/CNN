/*
This program uses N.E.A.T to teach an AI a linear algorithm with 3 inputs
*/

#include <CNN.h>
#include <neat.h>

int main()
{
    srand(time(NULL));
    uint16_t shape[]= {5,5,5,1};
    enum CNN_ActivationFunctions afs[] = {LeakyReLU, LeakyReLU,LeakyReLU,LeakyReLU};
    struct CNN_Structure structure = CNN_CreateStructure(sizeof(shape)/sizeof(uint16_t), 3, shape, afs);
    struct NEAT_Generation gen = NEAT_CreateGeneration(structure, 100);

    for (uint32_t o = 0; o < 30; o++) {    
        double input[3];
        randomArray(input, 3);
        double expectedOutput[] = {input[0] - input[1] + input[2]*.059 + 16};
        double* fitnesses = malloc(sizeof(double) * 100);
        double* costs = malloc(sizeof(double) * 100);
        for (uint32_t p = 0; p < 100; p++) {
            double* output = CNN_CalculateOutput(gen.population[p], input);
            fitnesses[p] = pow(0.9,fabs(output[0]-expectedOutput[0]));
            costs[p] = pow(output[0]-expectedOutput[0], 100);
        }
        NEAT_SimulationStep(gen, fitnesses, 1);
        double asdf = 0;
        for (uint16_t wer = 0; wer < 100; wer ++) {
            asdf += costs[wer]/100;
        }
        printf("%f\n", asdf);
    }

    return 0;
}