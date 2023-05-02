import numpy as np

# Genetic Algorithm
class genetic_algorithm:
    """Genetic algorithm optimization. This generates random set of individuals in a population given the possible gene values, computes fitness values of each, retains individual with highest fitness, and perform crossover to the least best and mutation to the worst fitnesses according to supplied probabilities. Then the process repeats itself according to the number of iterations.
    
    Parameters
    ----------
    gene_size : int
        Number of genes of each individual
    population_size : int
        Number of individuals in the population
    crossover_probability : float
        The probability of an individual's gene to cross over with that of the other individual / The number of genes that will be undergo crossover. If two genes will cross over, then the code picks up a random position in the individual's chromosome, chooses that gene, and skips one gene step to apply another crossover (Alternate crossover positions)
        Value must be from 0 - 100
    mutation_probability_gene : float
        Similar to crossover_probability but mutation will be performed. The type of mutation is through the use of uniform random distribution over given possible gene values
        Values must be from 0 - 100
    mutation_probability_population : float
        Probability of an individual to get mutated / Number of individuals to undergo mutation starting from the individual with worst fitness value
        Values must be from 0 - 100
    iterations : int
        Number of iterations
    fitness_function : function
        Function to be used to evaluate fitness values
    gene_values : array, list, tuple, etc.
        Possible values a gene takes
    optimization_type : str ('min' or 'max' only)
        Identifies if fitness_function is minimizing or maximizing
    record : Bool
            Default value : False
            If True, the run function records best fitness value for each iteration
    """
    
    def __init__(self,
                 gene_size, 
                 population_size, 
                 crossover_probability, 
                 mutation_probability_gene, 
                 mutation_probability_population, 
                 iterations, 
                 fitness_function, 
                 gene_values, 
                 optimization_type = 'min',
                 record = False):
        
        
        # Gene size
        self.n = gene_size
        
        # Population size
        self.N = population_size
        
        # Define iterations
        self.iterations = iterations
        
        # Define fitness function
        self.fitness_function = fitness_function
        
        # Define optimization type
        self.optimization_type = optimization_type
        
        # Define gene values
        self.gene_values = gene_values

        # Define record Bool value 
        self.record = record
        
        # Record populations per iteration
        self.container = []
        
        # Define population
        if len(self.gene_values) == 1:
            self.population = np.random.choice(gene_values[0], size = (self.N, self.n))
        
        else:
            self.population = np.zeros((self.N, self.n))
            for index1 in range(self.N):
                for index2, (i) in enumerate(self.gene_values):
#                 for index2 in range(self.n):
                    self.population[index1, index2] = np.random.choice(i)
#                 , size = (1, self.n))
#                 self.population = np.concatenate((self.population, 
#                                              np.random.choice(i, size = (self.N, 1))), axis = 1)
#             self.population = self.population[:, 1:]

        # Determine the number of individuals and
        # genes to perform crossover and mutation
        # For crossover
        self.cross_num = round(self.n * crossover_probability / 100)
        # Crossover step size, or the distance
        # between crossover genes
        self.cross_step = round(1 / crossover_probability)
        # For mutation of individuals
        self.mut_num1 = round(self.N * mutation_probability_population / 100)
        # For mutation of genes
        self.mut_num2 = round(self.n * mutation_probability_gene / 100)
        # Mutation step size
        self.mut_step = round(1 / mutation_probability_gene)

        # Number of individuals that will undergo crossover must be even
        self.cross_num_even = self.N - self.mut_num1 - 1
        if self.cross_num_even % 2 == 0:
            self.cross_num_even = self.cross_num_even
        else:
            self.cross_num_even = int(self.cross_num_even - 1)

        # Initializing container for recording best fit
        # for each iteration
        if self.record == True:
            self.best_fit = np.zeros(iterations)
        
    def run(self):
        """
        Returns
        -------
        Runs the genetic algorithm to determine solution to optimization problem
        """
        
        # Initialize first iteration
        iteration = 0
        # Perform iteration
        while iteration < self.iterations:
            # Evaluate fitnesses of each individual in the population
            if self.n == 1:
                fitnesses = self.fitness_function(self.population)
                fitnesses = fitnesses.reshape(1, self.N)[0]
            else:
                fitnesses = np.array(list(map(self.fitness_function, self.population)))
            # Check if optimzation is minimization or maximization
            if self.optimization_type == 'min':
                # Sort fitness values in descending order
                fit_args = np.argsort(fitnesses)
                # Record best fit for this iteration
                if self.record == True:
                    self.best_fit[iteration] = np.min(fitnesses)
                    self.container.append(self.population[0])
            elif self.optimization_type == 'max':
                # Sort fitness values in descending order
                fit_args = np.flip(np.argsort(fitnesses))
                # Record best fit for this iteration
                if self.record == True:
                    self.best_fit[iteration] = np.max(fitnesses)
                    self.container.append(self.population[0])
            
            # Arrange population in descending order of fitness values
            self.population = self.population[fit_args]
            
            # Update iterations
            iteration += 1

            # Apply crossover and mutation processes
            
            # Copy original population for easier crossover
            self.population_copy = np.copy(self.population)
            # Crossover
            for index in np.arange(1, 1 + self.cross_num_even, 2):
                # Random gene index for crossover
                random_index = np.random.randint(0, self.n)
                # Apply crossover
                for gene in range(0, self.cross_num):
                    self.population[index % self.N, (random_index + gene * self.cross_step) % self.n] = self.population_copy[(index + 1) % self.N, (random_index + gene * self.cross_step) % self.n]
                    self.population[(index + 1) % self.N, (random_index + gene * self.cross_step) % self.n] = self.population_copy[index % self.N, (random_index + gene * self.cross_step) % self.n]
                    
            # Apply mutation
            for index in range(1 + self.cross_num_even, self.N):
                # Random gene index for mutation
                random_index = np.random.randint(0, self.n)
                for gene in range(0, self.mut_num2):
                    if len(self.gene_values) == 1:
                        self.population[index, (random_index + gene * self.mut_step) % self.n] = np.random.choice(self.gene_values[0])
                    else:
                        gene_index = (random_index + gene * self.mut_step) % self.n
                        self.population[index, gene_index] = np.random.choice(self.gene_values[gene_index])
                        
    def best_values(self):
        """
        Returns
        -------
        Best values
        """
        return self.population[0]
    
    def code_efficiency(self):
        """
        Returns
        -------
        Fitness values over the nth iteration
        """
        if self.record == True:
            range_iterations = range(0, self.iterations)
            return range_iterations, self.best_fit
        else:
            print("Recording is not activated.")