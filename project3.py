"""
Put your documentation here!
"""

import operator, random, math

MAX_FLOAT = 10e12

def safe_division(numerator, denominator):
    """Divides numerator by denominator. If denominator is 0, returns
    MAX_FLOAT as an approximate of infinity."""
    if denominator == 0:
        return MAX_FLOAT
    return numerator / denominator

def safe_exp(power):
    """Takes e^power. If this results in a math overflow, instead returns
    MAX_FLOAT"""
    try:
        return math.exp(power)
    except OverflowError:
        return MAX_FLOAT

FUNCTION_DICT = {"+": operator.add,
                 "-": operator.sub,
                 "*": operator.sub,
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


def random_terminal():
    """Returns a random terminal node."""

    # Half of the time pick a variable, the other half pick a random
    # float in the range [-10, 10]
    if random.random() < 0.5:
        terminal_value = random.choice(VARIABLES)
    else:
        terminal_value = random.uniform(-10, 10)

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

        # Calculate values of children nodes
        children_results = [child.eval(variable_assignments) for child in self.children]

        # Apply function to children_results. * unpacks the list of results into
        # arguments to self.function.
        return self.function(*children_results)

    def tree_depth(self):
        """Returns the total depth of tree rooted at this node"""
        children_depths = [child.tree_depth() for child in self.children]
        return 1 + max(children_depths)


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




def main():

    # This program represents (+ (* x 5) y)
    program = FunctionNode("+",
                [FunctionNode("*",
                   [TerminalNode("x"),
                    TerminalNode(5.0)]),
                 TerminalNode("y")])

    print("Program:", program)
    print("Depth:", program.tree_depth())

    assignments = {"x": 7.0, "y": 9.0}

    print("program({}) =".format(assignments), program.eval(assignments))

    assignments = {"x": 3.0, "y": 1000.0}

    print("program({}) =".format(assignments), program.eval(assignments))
    print()

    # Make a full tree with depth = 4
    prog2 = GPNode.generate_tree_full(4)
    print(prog2)

    assignments = {"x": 7.0, "y": 9.0}

    print("prog2({}) =".format(assignments), prog2.eval(assignments))
    print()

    # Test 40 random programs to make sure no errors
    for _ in range(40):
        prog3 = GPNode.generate_tree_grow(6)
        print(prog3)
        print("prog3({}) =".format(assignments), prog3.eval(assignments))
        print()


if __name__ == "__main__":
    main()
