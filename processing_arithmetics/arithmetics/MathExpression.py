from __future__ import print_function
import numpy as np
import operator
import copy
import re
from collections import defaultdict, OrderedDict
from nltk import Tree
from numpy import random as random

class MathExpression(Tree):
    @classmethod
    def generateME(cls,length, operators, digits, branching=None):
        if length < 1: print('whatup?')
        elif length == 1:
            this = cls(random.choice(digits),[])
        else:
            if branching == 'left':
                left, right = length-1, 1
            elif branching == 'right':
                left, right = 1, length-1
            else:
                left = random.randint(1, length)
                right = length - left
            children = [cls.generateME(length=l,operators=operators, digits=digits, branching=branching) for l in [left,right]]
            children.insert(1,cls(np.random.choice(operators), []))
            this = cls('dummy', children)
        return this


    def __init__(self, label, children):
        if True: #len(children)>1:
            Tree.__init__(self, label, children)
            if len(children) > 1:
                self.maxDepth = max([child.maxDepth for child in children]) + 1
                self.length = sum([child.length for child in children])
            else:
                self.maxDepth = 0
                self.length = 1

        else:
            Tree.__init__(self, '',)
            self.maxDepth = 0
            self.length = 1

    @classmethod
    def fromstring(cls, string_repr):
        """
        Generate arithmetic expression from string.
        """
        list_repr = string_repr.split()
        nltk_list = []
        for symbol in list_repr:
            nltk_list.append(symbol)
            if symbol == '(':
                nltk_list.append('dummy')

        nltk_str = ' '.join(nltk_list)

        tree = Tree.fromstring(nltk_str)
        return cls.fromTree(tree)

    @classmethod
    def fromTree(cls, tree):
        if type(tree) is Tree:
            children = [cls.fromTree(c) for c in tree]
            return cls(tree.label(), children)
        else: return cls(tree,[])

    def property(self, propname):
        if propname == 'length': return self.length
        elif propname == 'maxDepth': return self.maxDepth
        elif propname == 'accumDepth': return self.length-1  #number of left brackets, len(re.findall('\(',str(self)))
        else: raise KeyError(propname+' is not a valid property of MathExpression')

    def solve(self):
        """
        Evaluate the expression
        """
        return eval(self.__str__())


    def toString(self, format='infix', digit_noise=None, operator_noise=None):
        """
        :param numbers noise: standard deviation of digit noise
        :param operator_noise: change of changing operator
        """
        operators = ['+','-']
        if self.label() == 'dummy':
            childrenStr = [c.toString(format,digit_noise,operator_noise) for c in self]
            if format == 'infix': return '( '+' '.join([childrenStr[0],childrenStr[1]]+childrenStr[2:])+' )'
            elif format == 'prefix': return '( '+' '.join([childrenStr[1],childrenStr[0]] + childrenStr[2:])+' )'
            elif format == 'postfix': return '( '+' '.join([childrenStr[0]]+childrenStr[2:] + [childrenStr[1]])+' )'
            else: raise ValueError("%s Unexisting format" % format)
        else:
            if self.label() in operators:
                if operator_noise:
                    del operators[operators.index(self.label())]
                    return (self.label() if np.random.uniform() < operator_noise else np.random.choice(operators))
                else: return self.label()
            else:
                if digit_noise > 0:
                    return str(np.random.normal(loc=int(self.label()), scale=digit_noise))
                else: return str(self.label())

    def __str__(self):
        """
        Return string representation of tree.
        """
        return self.toString()

    def solveRecursively(self, format='infix', return_sequences=False):
        """
        Solve expression recursively.
        """

        symbols = self.iterate(format)

        if format == "infix":
            return self.recursively_infix(symbols=symbols, return_sequences=return_sequences)

        elif format == "prefix":
            return self.recursively_prefix(symbols=symbols, return_sequences=return_sequences)

        elif format == "postfix":
            return self.recursively_postfix(symbols=symbols, return_sequences=return_sequences)
        
        else:
            assert ValueError("Invalid postfix")

    def recursively_infix(self, symbols, return_sequences=False):
        """
        Solve recursively for infix operator.
        """
        stack = []
        op = operator.add
        cur = 0

        # return arrays
        stack_list = []
        intermediate_results = []
        operators = []


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
            # store state

            if stack == []:
                stack_list.append([(-12345, -12345)])     # empty stack representation
            else:
                stack_list.append(copy.copy(stack))

            intermediate_results.append(cur)
            operators.append({operator.add:True, operator.sub: False}[op])

        assert len(stack) == 0, "expression not grammatical"

        if return_sequences:
            return intermediate_results, stack_list, operators

        return cur

    def recursively_prefix(self, symbols, return_sequences=False):
        operator_stack = []
        number_stack = []
        cur = 0
        intermediate_results, stack_list, operator_list = [], [], []

        for symbol in symbols:
            if symbol in ['+', '-']:
                op = {'+':operator.add, '-':operator.sub}[symbol]
                operator_stack.append(op)
            elif symbol == '(':
                pass
            elif symbol == ')':
                op = operator_stack.pop()
                prev = number_stack.pop()
                cur = op(prev, cur)
            else:
                # is digit
                digit = int(symbol)
                number_stack.append(cur)
                cur = digit

            intermediate_results.append(cur)
            # operator_list.append[{operator.add:True, operator.sub: False}[op]]

        if return_sequences:
            return intermediate_results, stack_list, operator_list

        return cur

    def recursively_postfix(self, symbols, return_sequences=False):

        stack = []
        cur = 0
        stack_list, intermediate_results, operator_list = [], [], []
        op = None

        for symbol in symbols:
            if symbol in ['+', '-']:
                op = {'+':operator.add, '-':operator.sub}[symbol]
                prev = stack.pop()
                cur = op(prev, cur)
            elif symbol in ['(', ')']:
                pass
            else:
                # is digit
                stack.append(cur)
                cur = int(symbol)

            if stack == []:
                stack_list.append([(-12345, -12345)])
            else:
                stack_list.append(copy.copy(stack))

            intermediate_results.append(cur)

        if return_sequences:
            return intermediate_results, stack_list, operator_list

        return cur
                

    def solveLocally(self, format='infix', return_sequences=False):
        """
        Input a syntactically correct bracketet
        expression, solve by counting brackets
        and depth.
        """

        symbols = self.iterate(format=format)

        if format == 'infix':
            return self.solve_locally_infix(symbols, return_sequences=return_sequences)

        elif format == 'prefix':
            return self.solve_locally_prefix(symbols, return_sequences=return_sequences)


    def solve_locally_infix(self, symbols, return_sequences=False):

        result = 0
        bracket_stack = []
        subtracting = False

        # return arrays
        intermediate_results = []
        brackets = []
        subtracting_list = []

        symbols = self.iterate(format='infix')

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
                subtracting = bracket_stack.pop(-1)
#                 bracket_stack.pop(-1)         git 
#                 try:
#                     subtracting = bracket_stack[-1]
#                 except IndexError:
#                     # end of sequence
#                     pass

            elif symbol == '+':
                pass

            elif symbol == '-':
                subtracting = not subtracting

            intermediate_results.append(result)
            brackets.append(bracket_stack)
            subtracting_list.append({True: [1], False:[0]}[subtracting])

        if return_sequences:
            return intermediate_results, brackets, subtracting_list
        
        else:
            return result

    def solve_locally_prefix(self, symbols, return_sequences=False):

        ops = {'+':operator.add, '-': operator.sub, False: operator.add, True: operator.sub}

        subtracting = False
        cur = 0
        stack = [False]

        for symbol in symbols:
            if symbol in ['+', '-']:
                sub = {'+': False, '-': True}[symbol]
                subtracting = stack.pop()
                if subtracting is False:
                    stack.append(sub)
                elif subtracting is True:
                    stack.append(not sub)
                stack.append(subtracting)

            elif symbol in ['(', ')']:
                pass

            else:
                # is digit
                digit = int(symbol)
                subtracting = stack.pop()
                op = ops[subtracting]
                cur = op(cur, digit)

        return cur

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

    
    def get_targets(self, format='infix'):
        """
        Compute all intermediate state variables
        that different approaches of computing the outcome
        of the equation would need.
        """
        intermediate_locally, brackets_locally, subtracting = self.solveLocally(return_sequences=True)
        intermediate_recursively, stack_recursively, subtracting_recursively = self.solveRecursively(return_sequences=True, format=format)

        self.targets = {}

        # grammaticality of sequence
        grammatical = [[0]]*len(intermediate_locally)
        grammatical[-1] = [1]
        self.targets['grammatical'] = grammatical

        # intermediate outcomes local computation
        self.targets['intermediate_locally'] = [[val] for val in intermediate_locally]

        # subtracting
        self.targets['subtracting'] = subtracting

        # intermediate outcomes recursive computation
        self.targets['intermediate_recursively'] = [[val] for val in intermediate_recursively]

        # element on top of stack
        # self.targets['top_stack'] = [[stack[-1][-1]] for stack in stack_recursively]


    def print_all_targets(self):
        """
        List all possible targets
        """
        for target in self.targets:
            print(target)


    def iterate(self, format):
        """
        Iterate over symbols in expression.
        """
        for symbol in self.toString(format=format).split():
            yield symbol

def make_noise_plots():
    import matplotlib.pylab as plt
    digits = np.arange(-5,5)
    languages = OrderedDict([('L1', 30), ('L2', 150), ('L3', 150), ('L4',150) , ('L5',150)])
    m = MathTreebank(languages=languages, digits=digits)
    sae = {}
    sse = {}
    mae = []
    mse = []
    for name, m in test_treebank(seed=5, languages=languages):
        sae[name] = 0
        sse[name] = 0
        for expression, answer in m.examples:
            results_locally = np.array([expression.solveLocally(format="infix", return_sequences=True)[0]])
            results_recursively = np.array([expression.solveRecursively(format="infix", return_sequences=True)[0]])

            sae[name] += np.mean(np.absolute(results_locally-results_recursively))
            sse[name] += np.mean(np.square(results_locally-results_recursively))

        mse.append(sse[name]/len(m.examples))
        mae.append(sae[name]/len(m.examples))

    fig, ax = plt.subplots()
    ax.plot(range(5), mse, label='mse')
    ax.plot(range(5), mae, label='mae')
    ax.set_xticklabels(languages.keys())
    plt.legend()
    plt.show()

def test_solve_locally(format, digits, operators):
    m = MathTreebank()
    for length in np.arange(3,10):
        examples = m.generate_examples(operators=ops, digits=digits, n=500, lengths=[length])
        incorrect = 0.0
        for expression, answer in examples:
            outcome = expression.solveLocally(format=format)
            if outcome != answer:
                incorrect += 1
                # print(expression, answer, outcome)
                # raw_input()

        print("percentage incorrect for length %i: %f" % (length, incorrect/50))

def test_solve_recursively(format, digits, operators):
    m = MathTreebank()
    for length in np.arange(3,10):
        examples = m.generate_examples(operators=ops, digits=digits, n=5000, lengths=[length])
        incorrect = 0.0
        for expression, answer in examples:
            outcome = expression.solveRecursively(format=format)
            if outcome != answer:
                incorrect += 1
                print(expression.toString(format), answer, outcome)
                raw_input()

        print("percentage incorrect for length %i: %f" % (length, incorrect/50))
