#include <CNN.h>
#include <neat.h>

int main()
{
    srand(time(NULL));
    uint16_t shape[]= {5,5,5,1};
    enum CNN_ActivationFunctions afs[] = {LeakyReLU, LeakyReLU,LeakyReLU,LeakyReLU};
    struct CNN_Structure structure = CNN_CreateStructure(sizeof(shape)/sizeof(uint16_t), 3, shape, afs);
    struct NEAT_Generation gen = NEAT_CreateGeneration(structure, 100);

    for (uint32_t o = 0; o < 100000; o++) {    
        double input[3];
        randomArray(input, 3);
        double expectedOutput[] = {input[0] + input[1] - input[2]*.001 + 3};
        double* fitnesses = malloc(sizeof(double) * 100);
        for (uint32_t p = 0; p < 100; p++) {
            double* output = CNN_CalculateOutput(gen.population[p], input);
            fitnesses[p] = exp(-pow((output[0]-expectedOutput[0]) * .2, 2));
        }
        NEAT_SimulationStep(gen, fitnesses, 0.01);
        printf("%f\n", fitnesses[0]); //WATCH HOW IT GOES UP TO 1!!!!
    }

    return 0;
}