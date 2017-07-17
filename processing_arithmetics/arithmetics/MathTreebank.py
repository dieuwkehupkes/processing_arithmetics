from .MathExpression import MathExpression
from numpy import random as random
import numpy as np
import re
from collections import defaultdict


def parse_language(language_str):
    """
    Give in a string for a language, return
    a tuple with arguments to generate examples.
    :return:    (#leaves, operators, branching)
    """
    # var initialisation
    operators = ['+', '-']

    # find # leaves
    nr = re.compile('[0-9]+')
    n = int(nr.search(language_str).group())

    # compile regular expressions
    R = re.compile('R')
    plusmin = re.compile('\+|-')
    branch = re.compile('left|right')

    # find if there are any root requirements
    root = R.search(language_str)
    if root:
        r = root.span()[0]
        root_branching = branch.search(language_str, r)
        if root_branching:
            root_branching = root_branching.group()
        root_operator = plusmin.search(language_str, r)
        if root_operator:
            root_operator = root_operator.group()
    else:
        r = len(language_str)
        root_operator, root_branching = None, None

    # find operators
    op = plusmin.search(language_str, 0, r)
    if op:
        operators = [op.group()]

    # find branchingness
    branching = branch.search(language_str)
    if branching:
        branching = branching.group()

    # check if there are any root requirements

    return [n], operators, branching, root_operator, root_branching


class MathTreebank():
    def __init__(self, languages={}, digits=[]):
        self.examples = []  # attribute containing examples of the treebank
        self.operators = set([])  # attribute containing operators in the treebank
        self.digits = set([])  # digits in the treebank
        for name, N in languages.items():
            lengths, operators, branching, root_operator, root_branching = parse_language(name)
            [self.operators.add(op) for op in operators]
            self.add_examples(digits=digits, operators=operators, 
                              branching=branching, lengths=lengths, n=N,
                              root_operator=root_operator, root_branching=root_branching)

    def generate_examples(self, operators, digits, branching=None,
                          root_branching=None, root_operator=None, 
                          min=-60, max=60, n=1000, lengths=range(1,6)):
        """
        :param operators:       operators to be used in
                                arithmetic expressions \in {+,-,\,*}
        :param digits:          range with digits (list)
        :param branching:       set to 'left' or 'right' to restrict branching of trees
        :param n:               number of sentences in tree bank
        :param min:             min outcome of the composition function
        :param max:             max outcome of the composition function
        :param lengths:         number of numeric leaves of expressions
        """
        examples = []
        digits = [str(i) for i in digits]
        self.digits = self.digits.union(set(digits))
        self.operators = self.operators.union(set(operators))
        while len(examples) < n:
            l = random.choice(lengths)
            tree = MathExpression.generateME(l, operators, digits, branching=branching, 
                                             root_branching=root_branching, root_operator=root_operator)
            answer = tree.solve()
            if answer is None:
                continue
            if not (min <= answer <= max):
                continue
            examples.append((tree,answer))
        return examples

    def add_examples(self, digits, operators=['+', '-'], branching=None,
                     root_branching=None, root_operator=None, 
                     min_answ=-60, max_answ=60, n=1000,
                     lengths=range(1, 6)):
        """
        Add examples to treebank.
        """
        self.examples += \
            self.generate_examples(operators=operators, digits=digits, branching=branching,
                                   root_branching=root_branching, root_operator=root_operator,
                                   min=min_answ, max=max_answ, n=n, lengths=lengths)


    def paired_examples(self):
        examples2 = self.examples[:]
        np.random.shuffle(examples2)
        return [(ex1[0],ex2[0],('<' if ex1[1] < ex2[1] else ('>' if ex1[1] > ex2[1] else '='))) for (ex1,ex2) in zip(self.examples,examples2)]

    def add_example_from_string(self, example):
        """
        Add a tree to the treebank from its string representation.
        """
        tree = MathExpression.fromstring(example)
        ans = tree.solve()
        self.examples.append((tree, ans))


    def write_to_file(self, filename):
        """
        Generate a file containing the treebank.
        Every tree element is separated by spaces, a tab
        separates the answer from the sentence. E.g
        ( ( 5 + 6 ) - 3 )   8
        """
        f = open(filename, 'wb')
        for expression, answer in self.examples:
            f.write(str(expression)+'\t'+str(answer[1])+'\n')
        f.close()


class IndexedTreebank(MathTreebank):
    def __init__(self, languages={}, digits=[]):
        self.index = {'length':defaultdict(list),'max_depth':defaultdict(list),'accum_depth':defaultdict(list)}
        MathTreebank.__init__(self,languages,digits)
        self.examples = tuple(self.examples)
        self.update_index()

    def update_index(self,from_point = 0,keys=[]):
        for i, (tree, label) in enumerate(self.examples[from_point:]):
            for key in (self.index.keys() if keys == [] else keys):
                value = tree.property(key)
                if i+from_point not in self.index[key][value]:
                    self.index[key][value].append(i+from_point)

    def add_examples(self, digits, operators=['+', '-'], branching=None, 
                     root_branching=None, root_operator=None, 
                     min_answ=-60, max_answ=60,
                     n=1000, lengths=range(1, 6)):
        from_point = len(self.examples)
        self.examples += \
            tuple(self.generate_examples(operators=operators, digits=digits, branching=branching,
                                         root_branching=root_branching,
                                         root_operator=root_operator,
                                         min=min_answ, max=max_answ, n=n, lengths=lengths))
        self.update_index(from_point)

    def get_examples_property(self, property):
        if property not in self.index.keys(): raise KeyError('not a valid property in this IndexedTreebank')
        else: return {k: self.examples[v] for k, v in self.index[property]}


    def get_examples_property_value(self, property,value):
        return self.get_examples_property(property)[value]
