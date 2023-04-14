import numpy as np
from typing import Callable, Tuple

class GeneticAlgorithm:
    """
    A class that implements a simple genetic algorithm.

    Parameters
    ----------
    population_size : int
        The number of individuals in the population.
    gene_length : int
        The length of each individual's genome.
    max_generations : int
        The maximum number of generations to run the algorithm.
    crossover_rate : float
        The probability that two parents will produce offspring.
    mutation_rate : float
        The probability that a single gene in an individual will mutate.
    elitism : bool
        Whether to use elitism (i.e., select the best individuals to survive to the next generation).
    elite_size : int
        The number of elite individuals to select if elitism is used.
    
    Attributes
    ----------
    population : numpy.ndarray
        An array representing the current population of individuals.
    fitness_values : numpy.ndarray
        An array containing the fitness values of each individual in the population.
    
    Methods
    -------
    initialize_population()
        Initializes the population with random genes.
    evaluate_population()
        Evaluates the fitness of each individual in the population.
    tournament_selection()
        Selects parents using tournament selection.
    single_point_crossover()
        Performs crossover on selected parents using a single-point crossover.
    bit_flip_mutation()
        Performs mutation on the population by flipping bits.
    elitism_selection()
        Selects the next generation using elitism and tournament selection.
    run_genetic_algorithm()
        Runs the genetic algorithm to find the optimal solution.
    
    """

    def __init__(
        self,
        population_size: int,
        gene_length: int,
        max_generations: int,
        crossover_rate: float,
        mutation_rate: float,
        elitism: bool,
        elite_size: int,
    ) -> None:
        self.population_size = population_size
        self.gene_length = gene_length
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.elite_size = elite_size
        self.population =  np.empty((population_size, gene_length), dtype=np.int8)
        self.fitness_values = np.empty(population_size, dtype=np.float64)
    
    def initialize_population(self) -> None:
        """Initializes the population with random genes.

        Returns
        -------
        None

        """
        self.population = np.random.randint(2, size=(self.population_size, self.gene_length))
        
    def evaluate_population(self, individuals: np.ndarray) -> np.ndarray:
        """Evaluates the fitness of each individual in the population.

        Parameters
        ----------
        individuals : np.ndarray
            The individuals, with one individual per row.

        Returns
        -------
        np.ndarray
            An array of fitness values for the correnponding individual.
        """
        
        fitness_values = np.apply_along_axis(self.fitness_function, 1, individuals)
        return fitness_values        
    
    def tournament_selection(self, tournament_size) -> Tuple[np.ndarray, np.ndarray]:
        """Selects parents using tournament selection.

        Parameters
        ----------
        tournament_size : int
            The number of individuals in each tournament.

        Returns
        -------
        numpy.ndarray
            An array of selected individuals.
        numpy.ndarray
            An array of selected individuals.
        """
        num_individuals = self.population.shape[0]
        num_genes = self.population.shape[1]
        selected_individuals = np.empty((num_individuals, num_genes), dtype=np.int8)
        selected_individuals_fitness = np.empty(num_individuals, dtype=int)
        for i in range(num_individuals):
            # Select a random subset of the population to form the tournament
            tournament_indices = np.random.choice(num_individuals, tournament_size, replace=False)
            tournament_fitness = self.fitness_values[tournament_indices]
        
        # Select the individual with the highest fitness in the tournament
            tournament_winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected_individuals_fitness[i] = np.max(tournament_fitness)
            selected_individuals[i, :] = self.population[tournament_winner_index, :]   
        return selected_individuals, selected_individuals_fitness
    
    def single_point_crossover(self, individuals: np.ndarray)-> np.ndarray:
        """Perform single-point crossover on a set of parent individuals.

        Parameters
        ----------
        individuals : np.ndarray
            The parent individuals, with one individual per row.

        Returns
        -------
        np.ndarray
            The offspring individuals, with one individual per row.
        """
        
        num_individuals = individuals.shape[0]
        num_genes = individuals.shape[1]

        # Initialize offspring array with zeros
        offspring = np.empty((num_individuals, num_genes), dtype=np.int8)

        for i in range(0, num_individuals, 2):
            if np.random.rand() < self.crossover_rate:
                # Select crossover point at random
                crossover_point = np.random.randint(1, num_genes)

                # Perform single-point crossover
                offspring[i, :crossover_point] = individuals[i, :crossover_point]
                offspring[i, crossover_point:] = individuals[i+1, crossover_point:]
                offspring[i+1, :crossover_point] = individuals[i+1, :crossover_point]
                offspring[i+1, crossover_point:] = individuals[i, crossover_point:]
            else:
                # If no crossover is performed, simply copy parents to offspring
                offspring[i, :] = individuals[i, :]
                offspring[i+1, :] = individuals[i+1, :]

        return offspring
    
    def bit_flip_mutation(self, individuals: np.ndarray)-> np.ndarray:
        """Perform bit-flip mutation on a set of individuals.

        Parameters
        ----------
        individuals : np.ndarray
            The individuals to mutate, with one individual per row.

        Returns
        -------
        np.ndarray
            The mutated individuals
        """
        
        mask = np.random.rand(*individuals.shape) < self.mutation_rate
        individuals[mask] = 1 - individuals[mask]
        return individuals
    
    def elitism_selection(self, individuals: np.ndarray, fitness_values: np.ndarray)-> np.ndarray:
        """Select the top individuals as elites for the next generation.

        Parameters
        ----------
        individuals : np.ndarray
            _description_
        fitness_values : np.ndarray
            _description_

        Returns
        -------
        np.ndarray
           The elite individuals.
        """
                
        # Sort the population by fitness and truncate to the elite size
        elite_indices = np.argsort(fitness_values)[:self.elite_size-1:-1]

        # Select the top individuals as elites
        elites = individuals[elite_indices, :]

        return elites
    
    def run_genetic_algorithm(self, fitness_function: Callable, tournament_size: int = 3)-> None:
        """Run the genetic algorithm

        Parameters
        ----------
        fitness_function : Callable
            Funtion to optimize
        tournament_size : int, optional
            The number of individuals in each tournament (default is 3).
        
        Returns
        -------
        None
        
        """
        self.fitness_function = fitness_function
                
        # Initialize the population and fitness values
        self.initialize_population()

        for generation in range(self.max_generations):
            # Select parents and perform crossover
            parents, parents_fitness = self.tournament_selection(tournament_size)
            offspring = self.single_point_crossover(parents)
            
            # Mutate the children and evaluate the fitness of the new population
            offspring = self.bit_flip_mutation(offspring)
            offspring_fitness = self.evaluate_population(offspring)
        
            # Select the next generation
            if self.elitism:
                self.population = self.elitism_selection(np.vstack((parents, offspring)), 
                                                         np.concatenate((parents_fitness, offspring_fitness)))
            else:
                self.population = offspring
        
            self.fitness_values = self.evaluate_population(self.population)

            # Print the best fitness value for this generation
            print("Generation {}: Best Fitness = {}".format(generation, np.max(self.fitness_values)))

        # Return the best solution found
        best_index = np.argmax(self.fitness_values)
        self.best_individual = self.population[best_index]
    
if __name__=="__main__":
    import numpy as np

    # Define the fitness function to be maximized
    def onemax(x):
        return np.sum(x)
    
    # define the genetic algorithm parameters
    POPULATION_SIZE = 6
    GENE_LENGTH = 5
    MUTATION_RATE = 1.0 / GENE_LENGTH
    CROSSOVER_RATE = 0.8
    ELITISM = True
    ELITE_SIZE = 6
    MAX_GENERATIONS = 10
    
    ga = GeneticAlgorithm(population_size=POPULATION_SIZE,
                          gene_length=GENE_LENGTH,
                          max_generations=MAX_GENERATIONS,
                          crossover_rate=CROSSOVER_RATE,
                          mutation_rate=MUTATION_RATE,
                          elitism=ELITISM,
                          elite_size=ELITE_SIZE)
    
    ga.run_genetic_algorithm(fitness_function=onemax)
    print('Best solution: {}'.format(ga.best_individual))