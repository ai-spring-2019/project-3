"""
Your documentation here
"""

import dominion

import operator, random, math, copy, re
#import dill

CARDS = ["Province", "Gold", "Witch", "Market", "Laboratory", "Duchy", "CouncilRoom", "Smithy", "Moneylender", "Gardens", "Workshop", "Village", "Silver", "Estate", "Chapel", "Curse", "Copper"]

def erf_have_lt_y_x(y = None, x = None):
    """Returns the function symbol and function for this ERF"""

    if y == None:
        Y_RANGE = (1, 11)
        y = random.randint(Y_RANGE[0], Y_RANGE[1])

    if x == None:
        x = random.choice(CARDS)

    # The symbol for this function.
    symbol = "have_lt_{}_{}".format(y, x)

    def erf_fn(game, cards_owned):
        """A closure to return which child to choose depending on cards_owned.
        Every erf_fn needs to return the index of the child to choose.
        This one returns child 0 if you have less than y of card x, and
        child 1 otherwise."""

        number_of_x = cards_owned.get(x, 0)
        if number_of_x < y:
            return 0
        return 1

    return (symbol, erf_fn)


# We want ephemeral random functions (ERFs) that do things like the following:
# have_lt_y_x, where x is the name of a card, y is an integer. This would return
# the "true" branch (first child) if the player has less than y of card x. So,
# have_lt_3_Smithy is true if you have less than 3 Smithy.

FUNCTION_DICT = {"ERF have_lt_y_x": erf_have_lt_y_x}
FUNCTION_ARITIES = {"ERF have_lt_y_x": 2}
FUNCTIONS = list(FUNCTION_DICT.keys())

POPULATION_SIZE = 100
MAX_GENERATIONS = 20
TOURNAMENT_SIZE = 5


def random_terminal():
    """Returns a random terminal node.
    For Dominion, all terminals will be shuffled versions of the CARDS list"""

    terminal = copy.deepcopy(CARDS)
    random.shuffle(terminal)

    return TerminalNode(terminal)


class FunctionNode:
    """Internal nodes that contain Functions."""

    def __init__(self, function_symbol, children):
        if function_symbol.startswith("ERF"):
            # Handle ERFs
            erf_function = FUNCTION_DICT[function_symbol]
            self.function_symbol, self.function = erf_function()

        elif function_symbol.startswith("have_lt"):
            # Handle making a string have_lt_y_x function into the actual function
            match = re.search("have_lt_(\d+)_(.*)", function_symbol)
            y = int(match.group(1))
            x = match.group(2)
            self.function_symbol, self.function = erf_have_lt_y_x(y, x)

        else:
            # Handle normal functions
            self.function_symbol = function_symbol
            self.function = FUNCTION_DICT[self.function_symbol]

        self.children = children

    def __str__(self):
        result = "({}".format(self.function_symbol)
        for child in self.children:
            result += " " + str(child)
        result += ")"
        return result

    def eval(self, game, cards_owned):
        """Evaluates node given the game object and cards_owned by current
        player. You SHOULD NOT change game or cards_owned in any way (since
        otherwise you could just tell game to give you all the Provinces, otherwise
        known as cheating). But, you can use any information stored in them that
        would be accessible to players of the game -- in other words, you shouldn't
        be looking at your opponent's hand, the top card of your deck, or anything
        else that wouldn't be allowed."""

        # Since we are evolving decision trees, each FunctionNode should simply
        # pick between its children, returning one of them (evaluated)
        child_index_to_return = self.function(game, cards_owned)
        return self.children[child_index_to_return].eval(game, cards_owned)


    def tree_depth(self):
        """Returns the total depth of tree rooted at this node"""
        children_depths = [child.tree_depth() for child in self.children]
        return 1 + max(children_depths)

    def size_of_subtree(self):
        """Gives the size of the subtree of this node, in number of nodes."""
        children_sizes = [child.size_of_subtree() for child in self.children]
        return 1 + sum(children_sizes)


class TerminalNode:
    """Leaf nodes that contain terminals."""

    def __init__(self, terminal):
        self.terminal = terminal

    def __str__(self):
        return str(self.terminal)

    def eval(self, game, cards_owned):
        """Simply returns the terminal."""
        return self.terminal

    def tree_depth(self):
        """Returns the total depth of tree rooted at this node"""
        return 0

    def size_of_subtree(self):
        """Gives the size of the subtree of this node, in number of nodes. Since
        this is a terminal node, is always 1."""
        return 1


def generate_tree_full(max_depth):
    """Generates and returns a new tree using the Full method for tree
    generation and a given max_depth."""

    if max_depth == 0:
        return random_terminal()

    else:
        function_symbol = random.choice(FUNCTIONS)
        arity = FUNCTION_ARITIES[function_symbol]
        children = [generate_tree_full(max_depth - 1) for _ in range(arity)]

        return FunctionNode(function_symbol, children)

def generate_tree_grow(max_depth):
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
            children = [generate_tree_grow(max_depth - 1) for _ in range(arity)]

            return FunctionNode(function_symbol, children)

def initialize_tree(min_depth, max_depth):
    """Generates a tree using Full or Grow, with a depth somewhere between
    min_depth and max_depth inclusive"""
    depth = random.randint(min_depth, max_depth)
    if random.random() < 0.5:
        return generate_tree_full(depth)
    else:
        return generate_tree_grow(depth)

################################################################################
## Below here is a parser of evolved lisp programs. You can pass parse_lisp
## a string version of an evolved program and get back the program tree.
## You can therefore just store your best programs as strings, and use this
## to build them into programs to compare them, etc.

def get_tokens(lisp):
    """Given a string representation of Lisp code, break into tokens."""

    broken = lisp.split()

    tokens = []
    in_list = False
    building_list = ""

    for thing in broken:

        # Handle in a list
        if in_list:
            # Check if the end of the list
            if "]" in thing:
                tokens_ending_in_paren = []
                while thing[-1] == ")":
                    tokens_ending_in_paren.append(")")
                    thing = thing[:-1]

                building_list += " " + thing
                tokens_ending_in_paren.append(building_list)
                tokens_ending_in_paren.reverse()
                tokens += tokens_ending_in_paren

                in_list = False
                building_list = ""
            else:
                building_list += " " + thing

        # Handle start of lists like ['Gardens', 'Smithy', 'Gold', ...]
        elif thing[0] == "[":
            if thing[-1] == "]":
                tokens.append(thing)
            else:
                in_list = True
                building_list = thing

        # Handle parentheses
        elif thing[0] == "(":
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
        terminal = eval(token)
        return TerminalNode(terminal)

def parse_lisp(lisp):
    """Parses a string in lisp syntax into a GP program."""
    tokens = get_tokens(lisp)
    return build_syntax_tree(tokens)


################################################################################
## Classes for GP Players and Individuals

class GPPlayer(dominion.Player):
    """A player evolved through GP"""

    def __init__(self, gp_tree):
        dominion.Player.__init__(self)

        # This is where the AI's evolved strategy tree is stored
        self.strategy = gp_tree

    def choose_gain(self, game, coins, buys):
        """Uses self.strategy to find the buy order list, and then buys first
        card that can buy in the list."""

        # Since strategy is a GP tree, can just call with cards_owned
        buy_order_list = self.strategy.eval(game, self.cards_owned)

        # Find first card in list that can buy, and buy it
        for card_name in buy_order_list:
            if game.can_buy(card_name, coins):
                return card_name

        # If can't buy anything, pass
        return "pass"

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
        be a list of Dominion player trees to play as opponents.
        Plays 2 games against each opponent, one as player 0 and one as player 1."""

        # TMH CHANGE HERE

        self.errors = []

        # Self playing as Player 0
        for opponent in test_cases:
            player0 = GPPlayer(self.program)
            player1 = GPPlayer(opponent)
            verbose = False

            game = dominion.Dominion(player0, player1, verbose)
            winner = game.play()

            if winner == "draw":
                error = 0.5
            elif winner == 0:
                error = 0
            else:
                error = 1

            self.errors.append(error)

        # Self playing as Player 1
        for opponent in test_cases:
            player1 = GPPlayer(self.program)
            player0 = GPPlayer(opponent)
            verbose = False

            game = dominion.Dominion(player0, player1, verbose)
            winner = game.play()

            if winner == "draw":
                error = 0.5
            elif winner == 1:
                error = 0
            else:
                error = 1

            self.errors.append(error)

        print("ERRORS:", self.errors)

        self.total_error = sum(self.errors)

    def is_solution(self, threshold):
        """Returns True if total_error is less than threshold."""
        return self.total_error <= threshold

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
    new_subtree = initialize_tree(1, 4)

    # Replace the subtree and return the new program
    return replace_random_subtree(parent.program, new_subtree)

def crossover(parent1, parent2):
    """Crosses over two parents (individuals) to create a child program."""

    # Select a random subtree from parent2 to insert into parent1
    new_subtree = copy.deepcopy(random_subtree(parent2.program))

    # Replace the subtree and return the new program
    return replace_random_subtree(parent1.program, new_subtree)



def make_test_cases():
    """Makes a list of test cases. Each test case a random Dominion player tree."""

    NUM_CASES = 10
    cases = []

    for _ in range(10):
        tree = initialize_tree(2, 5)
        cases.append(tree)

    return cases

def report(generation, best_individual):
    """Prints a report for this generation."""

    print("===== Report at Generation {:3d} =====".format(generation))
    print("Best program: {}".format(best_individual.program))
    print("Best program size: {}".format(best_individual.nodes()))
    print("Best errors: {}".format(best_individual.errors))
    print("Best total error: {}".format(best_individual.total_error))
    print("====================================\n")

def gp():
    """Runs GP. Returns an individual with total_error of 0."""

    # Create test cases:
    test_cases = make_test_cases()

    # Create a population
    population = [Individual(initialize_tree(2, 5)) for _ in range(POPULATION_SIZE)]

    for generation in range(MAX_GENERATIONS):

        # Evaluate the population
        best_ind = population[0]
        for i, ind in enumerate(population):
            print("\nEvaluating individual {}".format(i))
            ind.evaluate_individual(test_cases)
            #print(ind)

            if ind.total_error < best_ind.total_error:
                best_ind = ind


        # Report about generation
        report(generation, best_ind)

        if best_ind.is_solution(0):
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

def play_games(ind0, ind1, num_games):
    """Plays num_games games between ind0 and ind1, returning a dictionary
    of the results."""
    record = {0:0, 1:0, "draw":0}

    for i in range(num_games):
        player0 = GPPlayer(ind0.program)
        player1 = GPPlayer(ind1.program)
        game = dominion.Dominion(player0, player1, False)
        winner = game.play()
        print(i, winner)
        record[winner] += 1

    return record


def main():

    cards_owned_example = {"Estate": 3,
                           "Copper": 7,
                           "Smithy": 8,
                           "Silver": 6,
                           "Village": 9,
                           "Gold": 12,
                           "Duchy": 4}

    #### This code creates a random tree and evaluates it with regards to some specific cards owned.
    # prog1 = initialize_tree(2, 5)
    # print(prog1)
    # print()
    # print(prog1.eval(None, cards_owned_example))


    ### These examples show the two ways to create ERFs, one with the parameters
    ### specified, and one random.
    # game = dominion.Dominion(dominion.Player(), dominion.Player(), False)
    # (sym, fn) = erf_have_lt_y_x()
    # print(sym, ",  child branch to choose =", fn(game, cards_owned_example))
    #
    # (sym, fn) = erf_have_lt_y_x(5, "Copper")
    # print(sym, ",  child branch to choose =", fn(game, cards_owned_example))


    ### This will run GP to evolve Dominion strategies
    # solution = gp()
    #
    # print("FINISHED GP")
    # print(solution)


    ### This shows how to dill (pickle) a solution program, which stores it in a binary
    ### file so that the object can be reloaded in its entirety. To use this,
    ### you will need to install dill with pip:
    ###   pip install dill
    ### and then uncomment the import to dill at the top of the program.
    ### You can read more about dill here: https://pypi.org/project/dill/
    ###
    ### (Note: If you have any experience with pickle and are wondering why I'm
    ### not using it here, it's because pickle can't handle closures like are
    ### used in the ERFs.)
    # file = open("solution.dat", "wb")
    # dill.dump(solution, file)
    # file.close()

    ### Let's say you've done this twice, and renamed the files "solution0.dat"
    ### and "solution1.dat". You can load them and have them play games against
    ### each other like this:
    # file0 = open("solution0.dat", "rb")
    # solution0 = dill.load(file0)
    # file0.close()
    # file1 = open("solution1.dat", "rb")
    # solution1 = dill.load(file1)
    # file1.close()
    #
    # print(solution0)
    # print(solution1)
    #
    # record = play_games(solution0, solution1, 100)
    #
    # print()
    # print(record)


    ### This shows how you can turn a program string into a program using parse_lisp
    ### Note: as-is, this only works with the ERFs have_lt_y_x. If you want to
    ### parse your own functions, you will need to add them to the constructor
    ### of FunctionNode. See the chunk of code starting with:
    ###        elif function_symbol.startswith("have_lt"):
    ### to see an example.
    # evolved_program_string = """(have_lt_10_Gold ['Gardens', 'Smithy', 'Gold', 'Village', 'Province', 'Copper', 'Moneylender', 'Market', 'Witch', 'Laboratory', 'Workshop', 'Chapel', 'Silver', 'CouncilRoom', 'Duchy', 'Estate', 'Curse'] (have_lt_11_Curse (have_lt_7_Silver ['Market', 'Workshop', 'Duchy', 'Laboratory', 'Gardens', 'Gold', 'Smithy', 'Witch', 'Estate', 'Copper', 'Province', 'Chapel', 'Moneylender', 'CouncilRoom', 'Curse', 'Silver', 'Village'] (have_lt_8_Smithy ['Village', 'Province', 'Workshop', 'Moneylender', 'Silver', 'Laboratory', 'Chapel', 'Witch', 'Duchy', 'Copper', 'Gardens', 'CouncilRoom', 'Gold', 'Smithy', 'Curse', 'Estate', 'Market'] ['Chapel', 'Laboratory', 'Workshop', 'Village', 'Silver', 'Market', 'Copper', 'Duchy', 'CouncilRoom', 'Gardens', 'Province', 'Smithy', 'Curse', 'Witch', 'Gold', 'Estate', 'Moneylender'])) ['Witch', 'Gardens', 'Village', 'Province', 'Chapel', 'Moneylender', 'CouncilRoom', 'Smithy', 'Market', 'Curse', 'Duchy', 'Workshop', 'Gold', 'Copper', 'Estate', 'Laboratory', 'Silver']))"""
    #
    # evolved_program = parse_lisp(evolved_program_string)
    # print(evolved_program)
    # print(evolved_program.eval(None, cards_owned_example))


if __name__ == "__main__":
    main()
