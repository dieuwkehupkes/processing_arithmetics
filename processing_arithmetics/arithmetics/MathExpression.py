from __future__ import print_function
import numpy as np
import operator
import re
from collections import defaultdict, OrderedDict
from nltk import Tree
from numpy import random as random

class MathExpression(Tree):
    @classmethod
    def generateME(cls, length, operators, digits, branching=None):
        if length < 1:
            print('This case should not happen')
        elif length == 1:
            this = cls(random.choice(digits), [])
        else:
            if branching == 'left':
                left, right = length-1, 1
            elif branching == 'right':
                left, right = 1, length-1
            else:
                left = random.randint(1, length)
                right = length - left
            children = [cls.generateME(length=l, operators=operators, digits=digits, branching=branching) for l in [left, right]]
            children.insert(1, cls(np.random.choice(operators), []))
            this = cls('dummy', children)
        return this


    def __init__(self, label, children):
        Tree.__init__(self, label, children)
        if len(children) > 1:
            self.max_depth = max([child.max_depth for child in children]) + 1
            self.length = sum([child.length for child in children])
        else:
            self.max_depth = 0
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
        return cls.from_tree(tree)

    @classmethod
    def from_tree(cls, tree):
        if type(tree) is Tree:
            children = [cls.from_tree(c) for c in tree]
            return cls(tree.label(), children)
        else: return cls(tree, [])

    def property(self, propname):
        if propname == 'length': return self.length
        elif propname == 'max_depth': return self.max_depth
        elif propname == 'accum_depth': return self.length-1  #number of left brackets, len(re.findall('\(', str(self)))
        else: raise KeyError(propname+' is not a valid property of MathExpression')

    def solve(self, digit_noise=None, operator_noise=None):
        """
        Evaluate the expression
        """
        return eval(self.__str__(digit_noise=digit_noise, operator_noise=operator_noise))


    def to_string(self, format='infix', digit_noise=None, operator_noise=None):
        """
        :param numbers noise: standard deviation of digit noise
        :param operator_noise: change of changing operator
        """
        operators = ['+', '-']
        if self.label() == 'dummy':
            children_str = [c.to_string(format, digit_noise, operator_noise) for c in self]
            if format == 'infix': return '( '+' '.join([children_str[0], children_str[1]]+children_str[2:])+' )'
            elif format == 'prefix': return '( '+' '.join([children_str[1], children_str[0]] + children_str[2:])+' )'
            elif format == 'postfix': return '( '+' '.join([children_str[0]]+children_str[2:] + [children_str[1]])+' )'
            else: raise ValueError("%s Unexisting format" % format)
        else:
            if self.label() in operators:
                if operator_noise:
                    del operators[operators.index(self.label())]
                    return (self.label() if np.random.uniform() > operator_noise else np.random.choice(operators))
                else: return self.label()
            else:
                if digit_noise > 0:
                    return str(np.random.normal(loc=int(self.label()), scale=digit_noise))
                else: return str(self.label())

    def __str__(self, digit_noise=None, operator_noise=None):
        """
        Return string representation of tree.
        """
        return self.to_string(digit_noise=digit_noise, operator_noise=operator_noise)

    def solve_recursively(self, format='infix', return_sequences=False, digit_noise=None, operator_noise=None, stack_noise=None):
        """
        Solve expression recursively.
        """

        symbols = self.iterate(format=format, digit_noise=digit_noise, operator_noise=operator_noise)

        if format == "infix":
            return self.recursively_infix(symbols=symbols, return_sequences=return_sequences, stack_noise=stack_noise)

        elif format == "prefix":
            return self.recursively_prefix(symbols=symbols, return_sequences=return_sequences, stack_noise=stack_noise)

        elif format == "postfix":
            return self.recursively_postfix(symbols=symbols, return_sequences=return_sequences)
        
        else:
            assert ValueError("Invalid postfix")

    def recursively_infix(self, symbols, return_sequences=False, stack_noise=None):
        """
        Solve recursively for infix operator.
        """
        op_dict = {-1: operator.sub, 1: operator.add}

        digit_stack, operator_stack = [], []
        op = 1
        result = 0

        # return arrays
        stack_list = []
        intermediate_results = []
        operators = []


        for symbol in symbols:
            if stack_noise:
                # apply noise to stack
                operator_stack = self.add_noise(operator_stack, stack_noise=stack_noise)
                digit_stack = self.add_noise(digit_stack, stack_noise=stack_noise)

            if symbol == '(':
                # push new element on stack
                operator_stack.append(op)
                digit_stack.append(result)
                op = 1
                result = 0         # reset current computation
            elif symbol == ')':
                # combine last stack item with
                # one but last stack item
                op = np.power(-1, np.floor(operator_stack.pop()/2))
                prev = digit_stack.pop()
                result = op_dict[op](prev, result)
            elif symbol == '+':
                op = 1
            elif symbol == '-':
                op = -1
            else:
                # number is digit
                result = op_dict[op](result, float(symbol))
            # store state

            if digit_stack == []:
                stack_list.append([0, 0])     # empty stack representation TODO change this
            else:
                stack_list.append([digit_stack[:], operator_stack[:]])

            intermediate_results.append(result)
            operators.append({1: True, -1: False}[op])

        assert len(digit_stack) == 0, "expression not grammatical"

        if return_sequences:
            return intermediate_results, stack_list, operators

        return result

    def recursively_prefix(self, symbols, return_sequences=False, stack_noise=None):
        operator_stack = []
        digit_stack = []
        result = 0
        intermediate_results, stack_list, operator_list = [], [], []

        op_dict = {-1: operator.sub, 1: operator.add}

        for symbol in symbols:
            if stack_noise:
                # apply noise to stack
                operator_stack = self.add_noise(operator_stack, stack_noise=stack_noise)
                digit_stack = self.add_noise(digit_stack, stack_noise=stack_noise)

            if symbol in ['+', '-']:
                op = {'+':1, '-':-1}[symbol]
                operator_stack.append(op)
            elif symbol == '(':
                pass
            elif symbol == ')':
                op = np.power(-1, np.floor(operator_stack.pop()/2))
                prev = digit_stack.pop()
                result = op_dict[op](prev, result)
            else:
                # is digit
                digit = float(symbol)
                digit_stack.append(result)
                result = digit

            intermediate_results.append(result)
            # operator_list.append[{operator.add:True, operator.sub: False}[op]]

        if return_sequences:
            return intermediate_results, stack_list, operator_list

        return result

    def recursively_postfix(self, symbols, return_sequences=False, stack_noise=None):

        stack = []
        result = 0
        stack_list, intermediate_results, operator_list = [], [], []
        op = None

        for symbol in symbols:
            if stack_noise:
                # apply noise to stack
                stack = self.add_noise(stack, stack_noise=stack_noise)
            if symbol in ['+', '-']:
                op = {'+':operator.add, '-':operator.sub}[symbol]
                prev = stack.pop()
                result = op(prev, result)
            elif symbol in ['(', ')']:
                pass
            else:
                # is digit
                stack.append(result)
                result = float(symbol)

            if stack == []:
                stack_list.append([])
            else:
                stack_list.append(stack[:])

            intermediate_results.append(result)

        if return_sequences:
            return intermediate_results, stack_list, operator_list

        return result
                

    def solve_locally(self, format='infix', return_sequences=False, digit_noise=None, operator_noise=None, stack_noise=None):
        """
        Input a syntactically correct bracketet
        expression, solve by counting brackets
        and depth.
        """

        symbols = self.iterate(format=format, digit_noise=digit_noise, operator_noise=operator_noise)

        if format == 'infix':
            return self.solve_locally_infix(symbols, return_sequences=return_sequences, stack_noise=stack_noise)

        elif format == 'prefix':
            return self.solve_locally_prefix(symbols, return_sequences=return_sequences, stack_noise=stack_noise)

        elif format == 'postfix':
            return None, None, None


    def solve_locally_infix(self, symbols, return_sequences=False, digit_noise=None, operator_noise=None, stack_noise=None):

        result = 0
        operator_stack = []
        op = 1

        # return arrays
        intermediate_results = []
        brackets = []
        operator_list = []
        op_dict = {-1: operator.sub, 1: operator.add}

        for symbol in symbols:
            if stack_noise:
                # apply noise to stack
                operator_stack = self.add_noise(operator_stack, stack_noise=stack_noise)
                # apply noise to memory
                op = op + np.random.normal(0, stack_noise)
                result = result + np.random.normal(0, stack_noise)

            if symbol[-1].isdigit():
                digit = float(symbol)
                result = op_dict[np.power(-1, np.floor(op/2))](result, digit)

            elif symbol == '(':
                operator_stack.append(op)

            elif symbol == ')':
                op = operator_stack.pop()

            elif symbol == '+':
                pass

            elif symbol == '-':
                op = - op

            if return_sequences:
                intermediate_results.append(result)
                brackets.append(operator_stack[:])
                operator_list.append({-1: 1, 1:0}[op])

        if return_sequences:
            return intermediate_results, brackets, operator_list
        
        else:
            return result

    def solve_locally_prefix(self, symbols, return_sequences=False, stack_noise=None):

        op_dict = {-1: operator.sub, 1: operator.add}

        op = 1
        result = 0
        stack = [1]

        # return arrays
        intermediate_results = []
        operators = []
        operator_list = []

        for symbol in symbols:
            if stack_noise:
                # apply noise to stack
                stack = self.add_noise(stack, stack_noise=stack_noise)

            if symbol in ['+', '-']:
                prev_op = np.power(-1, np.floor(stack.pop()/2))
                op = eval(symbol+'1')
                stack.append(op*prev_op)
                stack.append(prev_op)

            elif symbol in ['(', ')']:
                pass

            else:
                # is digit
                digit = float(symbol)
                op = np.power(-1, np.floor(stack.pop()/2))
                result = op_dict[op](result, digit)

            intermediate_results.append(result)
            operators.append(stack[:])
            operator_list.append({1:1, -1:0}[op])

        if return_sequences:
            return intermediate_results, operators, operator_list

        else:
            return result

    def solve_almost(self, format='infix', return_sequences=False, digit_noise=None, operator_noise=None, stack_noise=None):
        """
        Solve expression with a simpel completely 
        local strategy that almost always gives the
        right answer, but not always.
        """

        symbols = self.iterate(format='infix', digit_noise=digit_noise, operator_noise=operator_noise)
    
        result = 0
        subtracting = False
    
        for symbol in symbols:
            if symbol[-1].isdigit():
                digit = float(symbol)
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

    def solve_directly(self, format='infix', return_sequences=False, digit_noise=None, operator_noise=None, stack_noise=None):
        """
        Solve expression by just taking taking the value of every
        operator.
        """

        symbols = self.iterate(format='infix', digit_noise=digit_noise, operator_noise=operator_noise)

        # print(self.to_string())
    
        result = 0
        results = []
        op = operator.add
    
        for symbol in symbols:
            if symbol[-1].isdigit():
                digit = float(symbol)
                result = op(result, digit)
            elif symbol == '-':
                op = operator.sub
            elif symbol == '+':
                op = operator.add

            results.append(result)

        if return_sequences:
            return results

        return result

    def get_minus_depths(self, depth, counter=False, format='infix'):
        """
        Return a sequence that has value 1 for every
        point where the symbol is in a depth level deep
        embedded minus sequence.
        """
        assert format == 'infix', "minus depth non sensible target for %s" % format

        try:
            return self.depths[depth]
        except:
            pass

        symbols = self.iterate(format=format)

        l = len(str(self).split())

        self.depths = dict(zip([1, 2, 3, 4], np.zeros(4*l).reshape(4,l)))
        stack_depth = []
        self.depth_counts = dict(zip([1, 2, 3, 4], np.zeros(4*l).reshape(4,l)))
        expect_left = False

        i = 0
        for symbol in symbols:

            # check if scope has ended
            if stack_depth:
                if stack_depth[-1] == 0 and not expect_left:
                    stack_depth.pop(-1)

            if symbol == '-':
                stack_depth.append(0)
                expect_left = True

            elif stack_depth:
                if expect_left and symbol != '(':
                    expect_left = False

                elif symbol == '(':
                    stack_depth[-1] += 1

                elif symbol == ')':
                    if stack_depth[-1] == 0:
                        stack_depth.pop(-1)

                    if stack_depth:
                        stack_depth[-1] -= 1
                                
            # change values for cur depths and lower
            for d in xrange(1, len(stack_depth)+1):
                # raw_input('\n%s' % symbol)
                try:
                    self.depths[d][i] = 1
                    self.depth_counts[d][i] = sum(stack_depth[d-1:])
                except KeyError:
                    self.depths[d] = np.zeros(l)
                    self.depths[d][i] = 1
                    self.depth_counts[d] = np.zeros(l)
                    self.depths[d][i] = 0

            # print(stack_depth)

            i += 1

        if counter:
            return self.depth_counts[depth]

        return self.depths[depth]

            
    def get_depths(self, format='infix'):
        """
        Return a sequence with the depth of
        the tree at each point in time.
        """
        symbols = self.iterate(format=format)

        depth = 0
        depth_array = []

        for symbol in symbols:
            if symbol == '(':
                depth += 1
            elif symbol == ')':
                depth -= 1

            depth_array.append(depth)

        return depth_array

    def get_modes(self, format='infix'):
        """
        Return the sequences of modes the model
        goes through.
        """

        assert format == 'infix', "I did not implement this for formats other than infix"
        symbols = self.iterate(format=format)

        operator_stack = []
        op = 1
        operator_list = []

        for symbol in symbols:

            if symbol == '(':
                operator_stack.append(op)

            elif symbol == ')':
                op = operator_stack.pop()

            elif symbol == '-':
                op = - op

            operator_list.append({-1: '-', 1:'+'}[op])

        return operator_list


    def add_noise(self, stack, stack_noise):
        # check if stack is empty
        if len(stack) == 0:
            return stack[:]

        noise = np.random.normal(0, stack_noise, len(stack))
        noisy_stack = list(stack[:] + noise)

        return noisy_stack[:]
        
    def get_targets(self, format='infix', *classifiers):
        """
        Compute all intermediate state variables
        that different approaches of computing the outcome
        of the equation would need.
        """

        if not classifiers:
            classifiers = ['intermediate_locally', 'intermediate_recursively']
        # create target dict
        self.targets = {}

        if 'intermediate_locally' in classifiers or 'subtracting' in classifiers or 'switch_mode' in classifiers:
            intermediate_locally, brackets_locally, subtracting = self.solve_locally(return_sequences=True)

            switch_mode = np.array([0] + [0 if subtracting[i] == subtracting[i-1] else 1 for i in xrange(1, len(subtracting))])

            # intermediate outcomes incremental computation
            self.targets['intermediate_locally'] = [[val] for val in intermediate_locally]

            # subtracting
            self.targets['subtracting'] = [[val] for val in subtracting]
            
            # switch mode
            self.targets['switch_mode'] = [[val] for val in switch_mode]
        
        if 'intermediate_recursively' in classifiers or 'grammatical' in classifiers:
            intermediate_recursively, stack_recursively, subtracting_recursively = self.solve_recursively(return_sequences=True, format=format)

            # sequence grammaticality
            grammatical = [[0]]*len(intermediate_recursively)
            grammatical[-1] = [1]
            self.targets['grammatical'] = grammatical

            # intermediate outcomes recursive computation
            self.targets['intermediate_recursively'] = [[val] for val in intermediate_recursively]

        if 'depth' in classifiers:
            self.targets['depth'] = [[val] for val in self.get_depths()]

        if 'minus1depth' in classifiers:
            self.targets['minus1depth'] = [[val] for val in self.get_minus_depths(1)]

        if 'minus2depth' in classifiers:
            self.targets['minus2depth'] = [[val] for val in self.get_minus_depths(2)]

        if 'minus3depth' in classifiers:
            self.targets['minus3depth'] = [[val] for val in self.get_minus_depths(3)]

        if 'minus4depth' in classifiers:
            self.targets['minus4depth'] = [[val] for val in self.get_minus_depths(4)]

        if 'minus1depth_count' in classifiers:
            self.targets['minus1depth_count'] = [[val] for val in self.get_minus_depths(1, True)]


    def print_all_targets(self, format='infix'):
        """
        List all possible targets
        """
        self.get_targets(format)
        for target in self.targets:
            print(target)

    def iterate(self, format, digit_noise=None, operator_noise=None):
        """
        Iterate over symbols in expression.
        """

        for symbol in self.to_string(format=format, digit_noise=digit_noise, operator_noise=operator_noise).split():
            yield symbol

