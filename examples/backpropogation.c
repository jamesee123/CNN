#include <CNN.h>
#include <neat.h>

int main()
{
    srand(time(NULL));
    uint16_t shape[]= {5,5,5,1};
    enum CNN_ActivationFunctions afs[] = {LeakyReLU, LeakyReLU,LeakyReLU,LeakyReLU};
    struct CNN_AI ai = CNN_CreateAI(sizeof(shape)/sizeof(uint16_t), 3, shape, afs);

    for (uint32_t o = 0; o < 100000; o++) {    
        double input[3];
        randomArray(input, 3);
        double expectedOutput[] = {input[0] + input[1] - input[2]*.001 + 3};
        printf("%f\n", CNN_BackPropogate(ai, input, expectedOutput, .01)); //WATCH AS THE NUMBER GOES LOWER YAY!!!
    }

    return 0;
}