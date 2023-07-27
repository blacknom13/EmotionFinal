from NewtonFunctionFile import NewtonFunction


class FullFunction():
    functions = []
    ranges = []

    def __init__(self, functions, ranges):
        self.functions = functions
        self.ranges = ranges

    def fx(self, x):
        res = 0
        for i in range(len(self.ranges)):
            if self.ranges[i][0] <= x < self.ranges[i][1]:
                res = self.functions[i].fx(x)
                break
        return res

    def printComplexFunction(self):
        max_len = 0
        funcs = []
        for i in range(len(self.functions)):
            max_len = max(max_len, len(self.functions[i].printFunction()))
            funcs.append(self.functions[i].printFunction())
        for i in range(len(self.functions)):
            print(funcs[i], end="")
            print(" " * (max_len - len(funcs[i])),end=" ; ")
            func_range = str(self.ranges[i])
            print(func_range[:-1] + "[")
