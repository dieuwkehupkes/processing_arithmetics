from __future__ import print_function
import numpy as np
import operator
from nltk import Tree
#import random
from numpy import random as random
import copy
import re
from collections import defaultdict, OrderedDict

languages_train = {'L1':3000, 'L2': 3000, 'L4':3000, 'L5':3000, 'L7':3000}
languages_heldout = {'L3':500, 'L6':800, 'L8':800}
languages_test = OrderedDict([('L1', 50), ('L2', 500), ('L3', 1500), ('L4', 3000), ('L5', 5000), ('L6', 10000), ('L7', 15000), ('L8', 15000), ('L9', 15000), ('L9_left', 15000)])
ds = np.arange(-10,11)
ops = ['+','-']

def training_treebank(seed, languages=languages_train, digits=ds):
    np.random.seed(seed)
    m = mathTreebank(languages, digits=digits)
    return m


def heldout_treebank(seed, languages=languages_heldout, digits=ds):
    np.random.seed(seed)
    m = mathTreebank(languages, digits=digits)
    return m

def test_treebank(seed, languages=languages_test, digits=ds):
    np.random.seed(seed)
    for name, N in languages.items():
        yield name, mathTreebank(languages={name: N}, digits=digits)

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
        operators = ops

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
	examples2=self.examples[:]
	np.random.shuffle(examples2)
	self.pairedExamples=[(ex1[0],ex2[0],('<' if ex1[1]<ex2[1] else ('>' if ex1[1]>ex2[1] else '='))) for (ex1,ex2) in zip(self.examples,examples2)]


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

class indexedTreebank(mathTreebank):
    def __init__(self, languages={}, digits=[]):
        self.index={'length':defaultdict(list),'maxDepth':defaultdict(list),'accumDepth':defaultdict(list)}
        mathTreebank.__init__(self,languages,digits)
        self.examples = tuple(self.examples)
        self.updateIndex()

    def updateIndex(self,fromPoint = 0,keys=[]):
        for i, (tree, label) in enumerate(self.examples[fromPoint:]):
            for key in (self.index.keys() if keys ==[] else keys):
              value = tree.property(key)
              if i+fromPoint not in self.index[key][value]:
                  self.index[key][value].append(i+fromPoint)

    def add_examples(self, digits, operators=['+', '-'], branching=None, min_answ=-60, max_answ=60,
                     n=1000, lengths=range(1, 6)):
        fromPoint = len(self.examples)
        self.examples+=tuple(self.generateExamples(operators=operators, digits=digits, branching=branching,
                                               min=min_answ, max=max_answ, n=n, lengths=lengths))
        self.updateIndex(fromPoint)

    def getExamplesProperty(self, property):
        if property not in self.index.keys(): raise KeyError('not a valid property in this indexedTreebank')
        else: return {k: self.examples[v] for k, v in self.index[property]}


    def getExamplesPropertyValue(self, property,value):
        return self.getExamplesProperty(property)[value]

class mathExpression(Tree):
    def __init__(self, length, operators, digits, branching=None):
        if length < 1: print('whatup?')
        elif length == 1:
            try:
                Tree.__init__(self,'digit',[random.choice(digits)])
            except:
                Tree.__init__(self, 'operator', [random.choice(operators)])
            self.maxDepth = 0
        else:
            if branching == 'left':
                left, right = length-1, 1
            elif branching == 'right':
                left, right = 1, length-1
            else:
                left = random.randint(1, length)
                right = length - left
            children = [mathExpression(l,operators, digits, branching) for l in [left,right]]
            operator = random.choice(operators)
            children.insert(1, mathExpression(1, [operator], []))
            Tree.__init__(self,operator,children)
            self.maxDepth = max([child.maxDepth for child in children])+1
        self.length = length


    def property(self, propname):
        if propname=='length': return self.length
        elif propname == 'maxDepth': return self.maxDepth
        elif propname == 'accumDepth': return self.length-1 #number of left brackets, len(re.findall('\(',str(self)))
        else: raise KeyError(propname+' is not a valid property of mathExpression')

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
        string_repr = self.make_string(format=format)
        if digit_noise or operator_noise:
            symbol_list = string_repr.split()
            symbols = [self.add_noise(symbol, digit_noise, operator_noise) for symbol in symbol_list]
            string_repr = ' '.join(symbols)

        return string_repr
            
    def add_noise(self, item, digit_noise, operator_noise):
        """
        Add noise to item
        """
        if item in ['(', ')']:
            pass
        elif item == '+':
            if np.random.uniform() < operator_noise:
                item = '-'
        elif item == '-':
            if np.random.uniform() < operator_noise:
                item = '-'
        else:
            # item is digit
            item = str(int(np.round(np.random.normal(loc=int(item), scale=digit_noise))))

        return item

    def make_string(self, format='infix'):

        if len(self) > 1:
            children = [child.toString(format) for child in self]
            if format=='infix': return '( ' + ' '.join(children) + ' )'
            elif format == 'prefix': return '( ' + ' '.join([children[1], children[0],children[2]]) + ' )'
            elif format == 'postfix': return '( ' + ' '.join([children[0],children[2],children[1]]) + ' )'
            else:
                raise ValueError("%s Unexisting format" % format)
        else: return str(self[0])

    def __str__(self):
        """
        Return string representation of tree.
        """
        if len(self) > 1:
            return '( '+' '.join([str(child) for child in self])+' )'
        else:
            return str(self[0])


    def solveRecursively(self, format='infix', return_sequences=False):
        """
        Solve expression recursively.
        """

        symbols = self.iterate(format)

        if format=="infix":
            return self.recursively_infix(symbols=symbols, return_sequences=return_sequences)

        elif format=="prefix":
            return self.recursively_prefix(symbols=symbols, return_sequences=return_sequences)

        elif format=="postfix":
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

        assert len(stack) == 0, "expression not grammatical"

        if return_sequences:
            return intermediate_results, stack_list

        return cur

    def recursively_prefix(self, symbols, return_sequences=False):
        operator_stack = []
        number_stack = []
        op = None
        cur = 0
        intermediate_results, stack_list = [], []

        for symbol in symbols:
            if symbol in ['+', '-']:
                op = {'+':operator.add, '-':operator.sub}[symbol]
                operator_stack.append(op)
                number_stack.append(cur)
                stack.append(cur)
            elif symbol == '(':
                pass
            elif symbol == ')':
                op = operator_stack.pop
                prev = number_stack.pop
                cur = op(prev, cur)
            else:
                # is digit
                operator_stack.append(cur)
                cur = digit

            # store
            if stack == []:
                stack_list.append([(-12345, -12345)])     # empty stack representation
            else:
                stack_list.append(copy.copy(stack))

            if return_sequences:
                return intermediate_results, stack_list

            intermediate_results.append(cur)

        return cur

    def recursively_postfix(self, symbols, return_sequences=False):

        stack = []
        cur = 0
        stack_list = []
        intermediate_results = []

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
            return intermediate_results, stack_list

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
        intermediate_recursively, stack_recursively = self.solveRecursively(return_sequences=True, format=format)

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


if __name__ == '__main__':
    digits = np.arange(-5,5)
    languages = {'L5':10}
    m = mathTreebank(languages=languages, digits=digits)
    for expression, answer in m.examples:
        print(expression.toString())
        print(expression.toString(digit_noise=0.5, operator_noise=0.5))
        raw_input('\n')
    exit()
    #     expression.get_targets()
    #     print(expression)
    #     for target in expression.targets:
    #         print("\n%s: %s" % (target, expression.targets[target]))

    for length in np.arange(3,10):
        examples = m.generateExamples(operators=ops, digits=digits, n=5000, lengths=[length])
        incorrect = 0.0
        for expression, answer in examples:
            outcome = expression.solveLocally(format='infix')
            if outcome != answer:
                incorrect += 1
                print(expression, answer, outcome)
                raw_input()

        print("percentage incorrect for length %i: %f" % (length, incorrect/50))
