"""
Put your documentation here!
"""

import operator, random, math, copy

MAX_FLOAT = 1e12

def safe_division(numerator, denominator):
    """Divides numerator by denominator. If denominator is 0, returns
    MAX_FLOAT as an approximate of infinity."""
    if abs(denominator) <= 1 / MAX_FLOAT:
        return MAX_FLOAT
    return numerator / denominator

def safe_exp(power):
    """Takes e^power. If this results in a math overflow, or is greater
    than MAX_FLOAT, instead returns MAX_FLOAT"""
    try:
        result = math.exp(power)
        if result > MAX_FLOAT:
            return MAX_FLOAT
        return result
    except OverflowError:
        return MAX_FLOAT

FUNCTION_DICT = {"+": operator.add,
                 "-": operator.sub,
                 "*": operator.mul,
                 "/": safe_division,
                 "exp": safe_exp,
                 "sin": math.sin,
                 "cos": math.cos}
FUNCTION_ARITIES = {"+": 2,
                    "-": 2,
                    "*": 2,
                    "/": 2,
                    "exp": 1,
                    "sin": 1,
                    "cos": 1}
FUNCTIONS = list(FUNCTION_DICT.keys())

VARIABLES = ["x", "y"]
POPULATION_SIZE = 100
MAX_GENERATIONS = 20
TOURNAMENT_SIZE = 5


def random_terminal():
    """Returns a random terminal node."""

    # Half of the time pick a variable, the other half pick a random
    # float in the range [-10, 10]
    if random.random() < 0.5:
        terminal_value = random.choice(VARIABLES)
    else:
        terminal_value = random.uniform(-10, 10)
        #terminal_value = 1.0

    return TerminalNode(terminal_value)

class GPNode:
    """Represents nodes in a program's tree."""

    def __init__(self):
        pass

    @classmethod
    def generate_tree_full(cls, max_depth):
        """Generates and returns a new tree using the Full method for tree
        generation and a given max_depth."""

        if max_depth == 0:
            return random_terminal()

        else:
            function_symbol = random.choice(FUNCTIONS)
            arity = FUNCTION_ARITIES[function_symbol]
            children = [GPNode.generate_tree_full(max_depth - 1) for _ in range(arity)]

            return FunctionNode(function_symbol, children)

    @classmethod
    def generate_tree_grow(cls, max_depth):
        """Generates and returns a new tree using the Grow method for tree
        generation and a given max_depth."""

        if max_depth == 0:
            return random_terminal()

        else:

            # 3/4 of the time pick a function, 1/4 of the time pick a terminal
            if random.random() < 0.25:
                return random_terminal()

            else:
                function_symbol = random.choice(FUNCTIONS)
                arity = FUNCTION_ARITIES[function_symbol]
                children = [GPNode.generate_tree_grow(max_depth - 1) for _ in range(arity)]

                return FunctionNode(function_symbol, children)

    @classmethod
    def initialize_tree(cls, min_depth, max_depth):
        """Generates a tree using Full or Grow, with a depth somewhere between
        min_depth and max_depth inclusive"""
        depth = random.randint(min_depth, max_depth)
        if random.random() < 0.5:
            return cls.generate_tree_full(depth)
        else:
            return cls.generate_tree_grow(depth)

class FunctionNode(GPNode):
    """Internal nodes that contain Functions."""

    def __init__(self, function_symbol, children):
        self.function_symbol = function_symbol
        self.function = FUNCTION_DICT[self.function_symbol]
        self.children = children

    def __str__(self):
        result = "({}".format(self.function_symbol)
        for child in self.children:
            result += " " + str(child)
        result += ")"
        return result

    def eval(self, variable_assignments):
        """Evaluates node given a dictionary of variable assignments."""

        try:
            # Calculate values of children nodes
            children_results = [child.eval(variable_assignments) for child in self.children]

            # Apply function to children_results. * unpacks the list of results into
            # arguments to self.function.
            return self.function(*children_results)
        except ValueError as e:
            print("----------\nWeird value error:", e)
            print("Node causing it:", self)
            raise

    def tree_depth(self):
        """Returns the total depth of tree rooted at this node"""
        children_depths = [child.tree_depth() for child in self.children]
        return 1 + max(children_depths)

    def size_of_subtree(self):
        """Gives the size of the subtree of this node, in number of nodes."""
        children_sizes = [child.size_of_subtree() for child in self.children]
        return 1 + sum(children_sizes)


class TerminalNode(GPNode):
    """Leaf nodes that contain terminals."""

    def __init__(self, terminal):
        self.terminal = terminal

    def __str__(self):
        return str(self.terminal)

    def eval(self, variable_assignments):
        """Evaluates node given a dictionary of variable assignments."""

        if self.terminal in variable_assignments:
            return variable_assignments[self.terminal]

        return self.terminal

    def tree_depth(self):
        """Returns the total depth of tree rooted at this node"""
        return 0

    def size_of_subtree(self):
        """Gives the size of the subtree of this node, in number of nodes. Since
        this is a terminal node, is always 1."""
        return 1


def get_tokens(lisp):
    """Given a string representation of Lisp code, break into tokens."""

    broken = lisp.split()

    tokens = []
    for thing in broken:
        # Handle parentheses
        if thing[0] == "(":
            tokens.append("(")
            tokens.append(thing[1:])

        elif thing[-1] == ")":
            tokens_ending_in_paren = []
            while thing[-1] == ")":
                tokens_ending_in_paren.append(")")
                thing = thing[:-1]
            if thing != "":
                tokens_ending_in_paren.append(thing)
            tokens_ending_in_paren.reverse()
            tokens += tokens_ending_in_paren

        else:
            # Handle no parentheses
            tokens.append(thing)

    return tokens

def build_syntax_tree(tokens):
    """Bulds an AST/GP tree based on tokens."""

    # Check for recursive case
    if tokens[0] == "(":
        assert tokens[-1] == ")"
        assert tokens[1] in FUNCTIONS

        # Tokens for all arguments
        args_tokens = tokens[2:-1]

        # children will have actual nodes for children
        children = []
        while len(args_tokens) > 0:
            first_token = args_tokens.pop(0)
            if first_token != "(":
                # Handle non-paren arguments
                children.append(build_syntax_tree([first_token]))
            else:
                # Handle paren arguments. Need to find matching paren.
                first_arg_tokens = [first_token]
                opened = 1

                while opened > 0:
                    token = args_tokens.pop(0)
                    if token == "(":
                        opened += 1
                    elif token == ")":
                        opened -= 1
                    first_arg_tokens.append(token)

                children.append(build_syntax_tree(first_arg_tokens))

        return FunctionNode(tokens[1], children)

    else:
        # Base case is when we have no more parentheses
        # Should only be here if we have a variable or float
        assert len(tokens) == 1

        token = tokens[0]
        if token in VARIABLES:
            return TerminalNode(token)
        else:
            return TerminalNode(float(token))

def parse_lisp(lisp):
    """Parses a string in lisp syntax into a GP program."""
    tokens = get_tokens(lisp)
    return build_syntax_tree(tokens)


class Individual:
    """Represents a GP individual"""

    def __init__(self, program):
        self.program = program
        self.errors = None
        self.total_error = None

    def __str__(self):
        return """Individual with:
|- Program: {}
|- Total Error: {}
|- Errors: {}""".format(self.program, self.total_error, self.errors)

    def evaluate_individual(self, test_cases):
        """Evaluates the individual given a set of test cases. test_cases should
        be a list of input/output pairs (tuples) telling what output should be produced
        given each input. Inputs are themselves dictionaries of assignments to
        variable names (strings), and outputs are floats."""

        self.errors = []
        for (input, correct_output) in test_cases:
            program_output = self.program.eval(input)

            error = abs(program_output - correct_output)
            self.errors.append(error)

        self.total_error = sum(self.errors)

    def is_solution(self, threshold):
        """Returns True if total_error is less than threshold."""
        return self.total_error < threshold

    def nodes(self):
        """Number of nodes in the program of this individual."""
        return self.program.size_of_subtree()


def tournament_selection(population, tournament_size):
    """Selects an individual from the population using tournament selection
    with given tournament size."""

    best = random.choice(population)

    # Consider tournament_size random individuals, and pick the best one
    for _ in range(tournament_size - 1):
        ind = random.choice(population)

        if ind.total_error < best.total_error:
            best = ind

    return best

def subtree_at_index(node, index):
    """Returns subtree at particular index in this tree. Traverses tree in
    depth-first order."""

    if index == 0:
        return node

    # Subtract 1 for the current node
    index -= 1

    # Go through each child of the node, and find the one that contains this index
    for child in node.children:
        child_size = child.size_of_subtree()
        if index < child_size:
            return subtree_at_index(child, index)
        index -= child_size

    return "INDEX {} OUT OF BOUNDS".format(index)


def replace_subtree_at_index(node, index, new_subtree):
    """Replaces subtree at particular index in this tree. Traverses tree in
    depth-first order."""

    # Return the subtree if we've found index == 0
    if index == 0:
        return new_subtree

    # Subtract 1 for the current node
    index -= 1

    # Go through each child of the node, and find the one that contains this index
    for child_index in range(len(node.children)):
        child_size = node.children[child_index].size_of_subtree()
        if index < child_size:
            new_child = replace_subtree_at_index(node.children[child_index], index, new_subtree)
            node.children[child_index] = new_child
            return node
        index -= child_size

    return "INDEX {} OUT OF BOUNDS".format(index)




def random_subtree(program):
    """Returns a random subtree from given program, selected uniformly."""

    nodes = program.size_of_subtree()
    node_index = random.randint(0, nodes - 1)

    return subtree_at_index(program, node_index)

def replace_random_subtree(program, new_subtree):
    """Replaces a random subtree with new_subtree in program, with node to
    be replaced selected uniformly."""

    nodes = program.size_of_subtree()
    node_index = random.randint(0, nodes - 1)

    new_program = copy.deepcopy(program)

    return replace_subtree_at_index(new_program, node_index, new_subtree)



def mutation(parent):
    """Mutates an individual parent by replacing a random subtree with a randomly
    generated subtree."""

    # Make a new subtree with depth between 1 and 4
    new_subtree = GPNode.initialize_tree(1, 4)

    # Replace the subtree and return the new program
    return replace_random_subtree(parent.program, new_subtree)

def crossover(parent1, parent2):
    """Crosses over two parents (individuals) to create a child program."""

    # Select a random subtree from parent2 to insert into parent1
    new_subtree = copy.deepcopy(random_subtree(parent2.program))

    # Replace the subtree and return the new program
    return replace_random_subtree(parent1.program, new_subtree)



def make_test_cases():
    """Makes a list of test cases. Each test case is a tuple where the first
    element is a dictionary containing the x and y assignments, and the second
    element is the correct output. Hard coded for the function
       f(x,y) = (x * 5) + y """

    cases = []

    for x in range(-10, 11, 2):
        for y in range(-10, 11, 2):
            correct_output = float((x * 5) + y)
            input_output = ({"x": float(x), "y": float(y)}, correct_output)
            cases.append(input_output)

    return cases

def report(generation, best_individual):
    """Prints a report for this generation."""

    print("===== Report at Generation {:3d} =====".format(generation))
    print("Best program: {}".format(best_individual.program))
    print("Best program size: {}".format(best_individual.nodes()))
    print("Best errors: {}".format(best_individual.errors))
    print("Best total error: {}".format(best_individual.total_error))
    print("====================================\n")

def gp(threshold):
    """Runs GP. Returns an individual with total_error less than threshold."""

    # Create test cases:
    test_cases = make_test_cases()

    # Create a population
    population = [Individual(GPNode.initialize_tree(2, 5)) for _ in range(POPULATION_SIZE)]

    for generation in range(MAX_GENERATIONS):

        # Evaluate the population
        best_ind = population[0]
        for ind in population:
            ind.evaluate_individual(test_cases)
            #print(ind)

            if ind.total_error < best_ind.total_error:
                best_ind = ind


        # Report about generation
        report(generation, best_ind)

        if best_ind.is_solution(threshold):
            return best_ind

        # Create children
        old_population = population
        population = []

        for _ in range(POPULATION_SIZE):
            # Use 50% mutation, 50% crossover

            if random.random() < 0.5:
                parent = tournament_selection(old_population, TOURNAMENT_SIZE)
                child = mutation(parent)
            else:
                parent1 = tournament_selection(old_population, TOURNAMENT_SIZE)
                parent2 = tournament_selection(old_population, TOURNAMENT_SIZE)
                child = crossover(parent1, parent2)

            population.append(Individual(child))


    return "FAILURE"



def main():

    test_cases = make_test_cases()

    # This program represents (+ (* x 5) y)
    program = FunctionNode("+",
                [FunctionNode("*",
                   [TerminalNode("x"),
                    TerminalNode(5.0)]),
                 TerminalNode("y")])

    prog2 = FunctionNode("-",
                [FunctionNode("sin",
                    [FunctionNode("/",
                        [TerminalNode(1),
                         TerminalNode(2)])
                    ]),
                 FunctionNode("exp",
                    [TerminalNode("y")])
                    ])

    prog3 = parse_lisp("(+ -9.44869115097491 x)")

    #print(prog3)

    prog4 = parse_lisp("(+ (* (+ (/ x y) (sin x)) (+ (- x -7.583525807352453) (+ (- x x) x))) (+ -9.44869115097491 x))")

    print(prog4)
    print(prog4.eval({"x": 5, "y": 1}))

    #
    # print("Program:", program)
    # print("Depth:", program.tree_depth())
    #
    # assignments = {"x": 7.0, "y": 9.0}
    #
    # print("program({}) =".format(assignments), program.eval(assignments))
    #
    # assignments = {"x": 3.0, "y": 1000.0}
    #
    # print("program({}) =".format(assignments), program.eval(assignments))
    # print()
    #
    # # Make a full tree with depth = 4
    #prog2 = GPNode.generate_tree_full(4)
    # print(prog2)
    #
    # assignments = {"x": 7.0, "y": 9.0}
    #
    # print("prog2({}) =".format(assignments), prog2.eval(assignments))
    # print()

    # Test 400 random programs to make sure no errors
    # for _ in range(400):
    #     prog3 = GPNode.generate_tree_grow(6)
    #     print(prog3)
    #     print("prog3({}) =".format(assignments), prog3.eval(assignments))
    #     print()

    # for x in test_cases:
    #     print(x)

    # ind1 = Individual(program)
    # ind1.evaluate_individual(test_cases)
    # print(ind1)
    # print()
    # #
    # ind2 = Individual(prog2)
    # ind2.evaluate_individual(test_cases)
    # print(ind2)
    # print()
    #
    # print(ind2.nodes())
    # for index in range(ind2.nodes()):
    #     print(index, ":", subtree_at_index(ind2.program, index))

    # print(ind2.nodes())
    # q = TerminalNode("Z")
    # for index in range(ind2.nodes()):
    #     new_program = copy.deepcopy(ind2.program)
    #     print(index, ":", replace_subtree_at_index(new_program, index, q))

    # Test Crossover
    # print("PARENT1:", ind1.program)
    # print("PARENT2:", ind2.program)
    # print("---------")
    # for _ in range(10):
    #     print(" " * 8, crossover(ind1, ind2))

    # Test mutation
    # print("PARENT2:", ind2.program)
    # print("---------")
    # for _ in range(10):
    #     print(" " * 8, mutation(ind2))

    #for _ in range(500):


    #solution = gp(0.5)

    #print("FINISHED GP")
    #print(solution)



if __name__ == "__main__":
    main()
