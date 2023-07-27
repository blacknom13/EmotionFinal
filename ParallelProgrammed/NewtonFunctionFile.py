import pandas as pd


class NewtonFunction:
    abscissas = []
    final_poly_consts = []

    def __init__(self, file):
        x_s, y_s = self.extractCoordinates(file)

        self.abscissas = x_s
        self.final_poly_consts = self.extractConsts(x_s, y_s)

    def extractCoordinates(self, file):
        cvs = pd.read_csv(file)
        return cvs['X'].tolist(), cvs['Y'].tolist()

    def extractConsts(self, x_s, y_s):
        consts_list = []
        consts_list.append(y_s[0])
        for i in range(1, len(x_s) + 1):
            if not self.allEqual(y_s):
                if len(y_s) == 1:
                    break
                for j in range(len(y_s) - 1):
                    div_term = x_s[j + i] - x_s[j]
                    y_s[j] = (y_s[j + 1] - y_s[j]) / div_term
                consts_list.append(y_s[0])
                y_s.pop()
            else:
                break
        return consts_list

    def fx(self, x):
        res = 0
        partial_term = []
        for i in range(len(self.final_poly_consts)):
            temp = self.final_poly_consts[i]
            for j in range(len(partial_term)):
                temp *= partial_term[j]
            partial_term.append(x - self.abscissas[i])
            res += temp
        return res

    def printFunction(self):
        poly = ""
        partial_term = []
        for i in range(len(self.final_poly_consts)):
            if round(self.final_poly_consts[i], 2) != 0:
                poly += (" + " if self.final_poly_consts[i] > 0 else " - ")
                poly += str(format("%.2f" % abs(self.final_poly_consts[i])))
                if len(partial_term) > 0:
                    poly += " * "
                for j in range(len(partial_term)):
                    poly += "(x - " + str(partial_term[j]) + ")"
            partial_term.append(self.abscissas[i])
        return poly[1:]

    def allEqual(self, y_s):
        all = True
        for i in range(len(y_s) - 1):
            if y_s[i] != y_s[i + 1]:
                all = False
                break
        return all
