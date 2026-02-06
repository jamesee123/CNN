/*
This program uses back propagation to train a neural network to learn a linear function with 3 inputs.
*/

#include <CNN.h>
#include <time.h>
#include <neat.h>

int main()
{
    clock_t begin = clock();

    srand(time(NULL));
    uint16_t shape[]= {5,5,5,1};
    enum CNN_ActivationFunctions afs[] = {LeakyReLU, LeakyReLU,LeakyReLU,LeakyReLU};
    struct CNN_AI ai = CNN_CreateAI(sizeof(shape)/sizeof(uint16_t), 3, shape, afs);

    for (uint32_t o = 0; o < 1000; o++) {    
        double input[3];
        randomArray(input, 3);
        double expectedOutput[] = {input[0] + input[1] - input[2]*.001 + 3};
        CNN_BackPropogate(ai, input, expectedOutput, .1);
        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("%f\n", time_spent);
    }

    return 0;
}