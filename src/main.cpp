#include <cstdint>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <bits/stdc++.h>
#include <hip/hip_runtime.h>
#include <CLI/App.hpp>
#include <CLI/Formatter.hpp>
#include <CLI/Config.hpp>

#define BLOCK_SIZE 512
#define THREADS_PER_BLOCK 256
#define BOARD_SIZE 200

using namespace std;

template<uint16_t boardSize>
struct individual {
    uint16_t fitness;
    uint8_t queensPosition[boardSize];
};

const uint16_t targetFitness = ((BOARD_SIZE) * (BOARD_SIZE - 1)) / 2;

// Copied from HIP bit_extrack sample
#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }


/**
 * Handles initializing the population with random locations for the queens. 
 *
 * @param population The representation of the solutions to populate with random values
 * @param populationSize The number of individuals in the population
 */
template <uint16_t boardSize>
void initPopulation(individual<boardSize> *population, uint16_t populationSize) {
    for(uint16_t individualIndex = 0; individualIndex < populationSize; individualIndex++) {
        for(uint8_t rowIndex = 0; rowIndex < BOARD_SIZE; rowIndex++) {
            // A queen can show up anywhere on the row from index 0 up to the size of the board
            uint8_t randomQueenIndex = rand() % BOARD_SIZE;
            population[individualIndex].queensPosition[rowIndex] = randomQueenIndex;
            population[individualIndex].fitness = 0;
        }
    }
}

/**
 * Calculate the fitness for an individual. The fitness is the number of pairs of queens that
 * do not collide with each other. The max fitness is therefore the sum of the sequence from
 * 1...(n-1) = (n - 1)(n - 2)/2
 *
 * @param population The individuals to calculate the fitness for
 * @param populationSize The number of individuals
 */
__global__
template<uint16_t boardSize>
void calculateFitness(individual<boardSize> *population, uint16_t populationSize) {
    // Determine the individual to calculate the fitness for
    size_t individualIndex = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    

    // Make sure we are within the bounds of the population count
    if(individualIndex >= populationSize) { return; }

    // Assume max fitness, then decrease fitness based on collisions
    uint16_t fitness = targetFitness;
    #pragma unroll
    for(int rowIndex = 0; rowIndex < BOARD_SIZE; rowIndex++) {
        uint8_t queenPosition = population[individualIndex].queensPosition[rowIndex];

        // Check each other queen location and check for a collision
        #pragma unroll
        for(int positionCheck = rowIndex + 1; positionCheck < BOARD_SIZE - 1; positionCheck++) {
            uint8_t otherQueenPosition = population[individualIndex].queensPosition[positionCheck];

            // If the queens are in the same column, that is a collision
            if(queenPosition == otherQueenPosition) {
                fitness--;
            }

            // If the queens are diagonal from each other, that is also a problem
            if(abs(queenPosition - otherQueenPosition) == abs(rowIndex - positionCheck)) {
                fitness--;
            }
        }
    }
    // Update fitness
    population[individualIndex].fitness = fitness;
}

/**
 * Handles running the reproduction on the population. Assumes that the more fit
 * individuals are towards the start of the array and the less fit towards the end.
 * Once reproduction takes place, the population is updated with the next
 * generation. Currently the top 50% get to reproduce, this can be improved.
 * 
 * @param population The individuals to run reproduction on. Will be replaced with the next generation.
 * @param populationSize The number of individuals in the population.
 */
template <uint16_t boardSize>
void reproduction(individual<boardSize> *population, uint16_t populationSize) {
    individual<boardSize> nextGen[populationSize];

    for(int i = 0; i < populationSize; i++) {
        // Use elitist approach of simply having the top 50% reproduce
        individual<boardSize> firstParent = population[rand() % (populationSize / 2)];
        individual<boardSize> secondParent = population[rand() % (populationSize / 2)];

        individual<boardSize> child;
        // Top half from parent 1, bottom half from parent 2
        int midPoint = BOARD_SIZE / 2;
        for(int j = 0; j < midPoint; j++) {
            child.queensPosition[j] = firstParent.queensPosition[j];
            child.queensPosition[j + midPoint] = secondParent.queensPosition[j + midPoint];
        }

        // Random mutation chance
        if(rand() % 100 <= 5) {
            child.queensPosition[rand() % BOARD_SIZE] = rand() % BOARD_SIZE;
        }
        nextGen[i] = child;
    }

    // Copy over the next generation into the current generation
    memcpy(population, nextGen, sizeof(individual<boardSize>) * populationSize);
}

/**
 * Used to compare and sort individuals in decenting order based on fitness
 */
template<uint16_t boardSize>
bool compareIndividuals(individual<boardSize> first, individual<boardSize> second) {
    return first.fitness > second.fitness;
}

int main(int argc, char **argv) {
    // Get command line arguments
    CLI::App app("N-Queens Problem Solver");

    uint16_t populationSize = 1000;
    app.add_option("-p, --population", populationSize, "Number of individuals in each generation, defaults to 1000");

    uint16_t maxGenerations = -1;
    app.add_option("-m, --max", maxGenerations, "Maximum generations to run for, defaults to infinite");

    uint16_t boardSize = 15;
    app.add_option("-s, --size", boardSize, "Size of the board, default to 15");

    // Print some starting information
    cout << "Welcome to the N-Queens Solver" << endl;
    cout << "Population Size: " << populationSize << " Max Generations: " << maxGenerations;

    CLI11_PARSE(app, argc, argv);

    // Initialize randomization
    srand(time(NULL));

    // Initialize the popultation with initially random data
    individual<boardSize> *h_population = static_cast<individual*>(calloc(populationSize, sizeof(individual)));
    initPopulation(h_population, populationSize);

    // Initialize device variables
    individual<boardSize> *d_population;
    CHECK(hipMalloc(&d_population, populationSize * sizeof(individual)));

    for(uint16_t populationId = 0; 1; populationId++) {
        // Calculate the fitness of the population
        CHECK(hipMemcpy(d_population, h_population, populationSize * sizeof(individual), hipMemcpyHostToDevice));
        hipLaunchKernelGGL(calculateFitness, dim3(BLOCK_SIZE), dim3(THREADS_PER_BLOCK), 0, 0, d_population, populationSize);
        CHECK(hipMemcpy(h_population, d_population, populationSize * sizeof(individual), hipMemcpyDeviceToHost));

        // Sort the population based on fitness
        sort(h_population, h_population + populationSize, compareIndividuals);

        // Display the current highest fitness
        cout << "Generation: " << populationId << " best fitness: " << h_population[0].fitness << " target: " << targetFitness << endl;

        // Check to see if the end condition has been reached
        if(h_population[0].fitness >= targetFitness) {
            break;
        }

        // Run through reproduction
        reproduction(h_population, populationSize);
    }

    // Print if the ideal combination was found or not
    if(h_population[0].fitness >= targetFitness) {
        cout << "Ideal combination found!" << endl;
    } else {
        cout << "Could not find ideal combination" << endl;
    }

    // Print out the location to place the queens regardless of if it is the ideal configuration
    for(int i = 0; i < BOARD_SIZE; i++) {
        cout << +h_population[0].queensPosition[i] << " ";
    }
    cout << endl;

    // Free unused variables
    free(h_population);
    hipFree(d_population);

    return 0;
}
