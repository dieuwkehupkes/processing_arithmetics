from __future__ import print_function
import numpy as np
import operator
from nltk import Tree
import random
import re

def parse_language(language_str):
    """
    Give in a string for a language, return
    a tuple with arguments to generate examples.
    :return:    (#leaves, operators, branching)
    """
    # find # leaves
    nr = re.compile('[0-9]+')
    n = int(nr.search(language_str).group())

    # find operators
    plusmin = re.compile('\+')
    op = plusmin.search(language_str)
    if op:
        operators = [op.group()]
    else:
        operators = ['+','-']

    # find branchingness
    branch = re.compile('left|right')
    branching = branch.search(language_str)
    if branching:
        branching = branching.group()

    return [n], operators, branching

class mathTreebank():
    def __init__(self, languages={}, digits=[]):
        self.examples = []  # attribute containing examples of the treebank
        self.operators = set([])  # attribute containing operators in the treebank
        self.digits = set([])  # digits in the treebank
        for name, N in languages.items():
            lengths, operators, branching = parse_language(name)
            [self.operators.add(op) for op in operators]
            self.add_examples(digits=digits, operators=operators, branching=branching, lengths=lengths, n=N)



    def generateExamples(self, operators, digits, branching=None, min=-60, max=60, n=1000, lengths=range(1,6)):
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
            tree = mathExpression(l, operators, digits, branching=branching)
            answer = tree.solve()
            if answer is None:
                continue
            if not (min <= answer <= max):
                continue
            examples.append((tree,answer))
        return examples

    def add_examples(self, digits, operators=['+', '-'], branching=None, min_answ=-60, max_answ=60,
                     n=1000, lengths=range(1, 6)):
        """
        Add examples to treebank.
        """
        self.examples += self.generateExamples(operators=operators, digits=digits, branching=branching,
                                               min=min_answ, max=max_answ, n=n, lengths=lengths)

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


class mathExpression(Tree):
    def __init__(self, length, operators, digits, branching=None):
        if length < 1: print('whatup?')
        if length == 1:
            try:
                Tree.__init__(self,'digit',[random.choice(digits)])
            except IndexError:
                Tree.__init__(self, 'operator', [random.choice(operators)])
        else:
            if branching == 'left':
                left, right = length-1, 1
            elif branching == 'right':
                left, right = 1, length-1
            else:
                left = random.randint(1, length-1)
                right = length - left
            children = [mathExpression(l,operators, digits, branching) for l in [left,right]]
            operator = random.choice(operators)
            children.insert(1, mathExpression(1, [operator], []))
            Tree.__init__(self,operator,children)

    def solve(self):
        """
        Evaluate the expression
        """
        return eval(self.__str__())

    def __str__(self):
        """
        Return string representation of tree.
        """
        if len(self) > 1:
            return '( '+' '.join([str(child) for child in self])+' )'
        else:
            return self[0]


    def solveRecursively(self, return_sequences=False):
        """
        Solve expression recursively.
        """

        stack = []
        op = operator.add
        cur = 0

        symbols = self.iterate()

        # return arrays
        stack_list = []
        intermediate_results = []

        for symbol in symbols:
            if symbol == '(':
                # push new element on stack
                stack.append([op, cur])
                op = operator.add
                cur = 0         # reset current computation
            elif symbol == ')':
                # combine last stack item with
                # one but last stack item
                stack_op, prev = stack.pop()
                cur = stack_op(prev, cur)
            elif symbol == '+':
                op = operator.add
            elif symbol == '-':
                op = operator.sub
            else:
                # number is digit
                cur = op(cur, int(symbol))
            stack_list.append(stack)
            intermediate_results.append(cur)

        assert len(stack) == 0, "expression not grammatical"

        if return_sequences:
            return intermediate_results, stack_list

        return cur

    def solveLocally(self, return_sequences=False):
        """
        Input a syntactically correct bracketet
        expression, solve by counting brackets
        and depth.
        """
        result = 0
        bracket_stack = []
        subtracting = False

        # return arrays
        intermediate_results = []
        brackets = []

        symbols = self.iterate()

        for symbol in symbols:
            
            if symbol[-1].isdigit():
                digit = int(symbol)
                if subtracting:
                    result -= digit
                else:
                    result += digit

            elif symbol == '(':
                bracket_stack.append(subtracting)

            elif symbol == ')':
                bracket_stack.pop(-1)
                try:
                    subtracting = bracket_stack[-1]
                except IndexError:
                    # end of sequence
                    pass

            elif symbol == '+':
                pass

            elif symbol == '-':
                subtracting = not subtracting

            intermediate_results.append(result)
            brackets.append(bracket_stack)

        if return_sequences:
            return intermediate_results, brackets
        
        else:
            return result


    def solveAlmost(self, return_sequences=False):
        """
        Solve expression with a simpel completely 
        local strategy that almost always gives the
        right answer, but not always.
        """

        symbols = self.iterate()
    
        result = 0
        subtracting = False
    
        for symbol in symbols:
            if symbol[-1].isdigit():
                digit = int(symbol)
                if subtracting:
                    result -= digit
                else:
                    result += digit
            elif symbol == '-':
                subtracting = not subtracting
    
            if symbol == ')':
                if subtracting:
                    subtracting = False
    
        return result

    
    def get_targets(self):
        """
        Compute all intermediate state variables
        that different approaches of computing the outcome
        of the equation would need.
        """
        intermediate_locally, brackets_locally = self.solveLocally(return_sequences=True)
        sequences_recursively = self.solveRecursively(return_sequences=True)

        self.targets = {}

        # grammaticality of sequence
        grammatical = [[0]]*len(intermediate_locally)
        grammatical[-1] = [1]
        self.targets['grammatical'] = grammatical


        # intermediate outcomes local computation
        self.targets['intermediate_locally'] = [[val] for val in intermediate_locally]

        # TODO introduce more!!



    def iterate(self):
        """
        Iterate over symbols in expression.
        """
        for symbol in str(self).split():
            yield symbol


if __name__ == '__main__':
    m = mathTreebank()
    ops = ['+','-']
    digits = np.arange(-5,5)
    for length in np.arange(3,10):
        examples = m.generateExamples(operators=ops, digits=digits, n=5000, lengths=[length])
        incorrect = 0.0
        for expression, answer in examples:
            outcome = expression.solveRecursively()
            if outcome != answer:
                incorrect += 1

        print("percentage incorrect for length %i: %f" % (length, incorrect/50))
