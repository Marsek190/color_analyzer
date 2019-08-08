#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  12 11:03:49 2019

@author: jonik
"""

import numpy as np
import copy
import sys
import itertools
import scipy.stats as st
from functools import reduce
import operator
 
def is_empty(obj):
    return len(obj) == 0

"""
scope: globals(), locals(), __dict__, ...
"""
def compact(scope, *keys):
    return dict((k, scope[k]) for k in keys)

''''
normal dist class
extends at st.rv_continuous
'''
class NormalUniform(st.rv_continuous):
    def __init__(self, *args,  **kwargs):
        self._sigma = kwargs["sigma"]
        self._mean = kwargs["mean"]
        kwargs.clear()
        return super(NormalUniform, self).__init__(*args, **kwargs)
    
    def _pdf(self, x):
        D = 10 / np.log10(np.e)
        return np.sqrt(2 * np.pi * D * x * self._sigma) * np.exp(-(10 * np.log10(x) - 10 * np.log10(self._mean)) ** 2 / 2 * self._sigma ** 2)

'''
реализация венгерского алгоритма
'''
class Hungarian:

    SIMPLE = 0
    STARRED = 1
    PRIMED = 2
    
    def __init__(self):
        pass
        
    def maximize(self, cost_matrix):
        matrix = copy.deepcopy(cost_matrix)
        m = max(max(row) for row in matrix)
        for row in matrix:
            row[:] = list(map(lambda x: m - x, row))
    
        return self.minimize(matrix)
    
    
    def minimize(self, cost_matrix):        
        matrix = copy.deepcopy(cost_matrix)          
        n = len(matrix)

        for row in matrix:
            m = min(row)
            if m != 0:
                row[:] = list(map(lambda x: x - m, row))
    
        mask_matrix = [[self.SIMPLE] * n for _ in matrix]
        row_cover = [False] * n
        col_cover = [False] * n

        for r, row in enumerate(matrix):
            for c, value in enumerate(row):
                if value == 0 and not row_cover[r] and not col_cover[c]:
                    mask_matrix[r][c] = self.STARRED
                    row_cover[r] = True
                    col_cover[c] = True
    
        row_cover = [False] * n
        col_cover = [False] * n

        match_found = False
    
        while not match_found:
            for i in range(n):
                col_cover[i] = any(mrow[i] == self.STARRED for mrow in mask_matrix)
    
            if all(col_cover):
                match_found = True
                continue
            else:
                zero = self._cover_zeroes(matrix, mask_matrix, row_cover, col_cover)

                primes = [zero]
                stars = []
                while zero:
                    zero = self._find_star_in_col(mask_matrix, zero[1])
                    if zero:
                        stars.append(zero)
                        zero = self._find_prime_in_row(mask_matrix, zero[0])
                        stars.append(zero)

                for star in stars:
                    mask_matrix[star[0]][star[1]] = self.SIMPLE
    
                # отмечаем звездочкой
                for prime in primes:
                    mask_matrix[prime[0]][prime[1]] = self.STARRED
    
                # убираем отмеченные звездой элементы
                for r, row in enumerate(mask_matrix):
                    for c, val in enumerate(row):
                        if val == self.PRIMED:
                            mask_matrix[r][c] = self.SIMPLE
    
                row_cover = [False] * n
                col_cover = [False] * n

        solution = []
        for r, row in enumerate(mask_matrix):
            for c, val in enumerate(row):
                if val == self.STARRED:
                    solution.append((r, c))
        total_cost = self._sum_cost(solution, cost_matrix)    
        return total_cost, matrix
  
    def _cover_zeroes(self, matrix, mask_matrix, row_cover, col_cover):
        while True:
            zero = True
            while zero:
                zero = self._find_noncovered_zero(matrix, row_cover, col_cover)
                if not zero:
                    break
                else:
                    row = mask_matrix[zero[0]]
                    row[zero[1]] = self.PRIMED
    
                    try:
                        index = row.index(self.STARRED)
                    except ValueError:
                        return zero
  
                    row_cover[zero[0]] = True
                    col_cover[index] = False

            m = min(self._uncovered_values(matrix, row_cover, col_cover))
            for r, row in enumerate(matrix):
                for c, __ in enumerate(row):
                    if row_cover[r]:
                        matrix[r][c] += m
                    if not col_cover[c]:
                        matrix[r][c] -= m
    def _find_noncovered_zero(self, matrix, row_cover, col_cover):
        for r, row in enumerate(matrix):
            for c, value in enumerate(row):
                if value == 0 and not row_cover[r] and not col_cover[c]:
                    return (r, c)
        else:
            return None

    def _uncovered_values(self, matrix, row_cover, col_cover):
        for r, row in enumerate(matrix):
            for c, value in enumerate(row):
                if not row_cover[r] and not col_cover[c]:
                    yield value
    
    
    def _find_star_in_col(self, mask_matrix, c):
        for r, row in enumerate(mask_matrix):
            if row[c] == self.STARRED:
                return (r, c)
        else:
            return None
    
    def _find_prime_in_row(self, mask_matrix, r):
        for c, val in enumerate(mask_matrix[r]):
            if val == self.PRIMED:
                return (r, c)
        else:
            return None
        
    def _sum_cost(self, results, cost_matrix):
        return sum([cost_matrix[r][c] for r, c in results])

class Task:
    VISITED = -1
    INCORRECT = float('inf')
    
    def __init__(self, C, Q, identity, indexes, min_rel, minimal_freq):        
        self._identity = identity
        # исходные позиции единичек
        self._indexes = indexes
        # начальное кач-во связи в узлах
        self._ev_rel = min_rel
        # преобразованная матрица коэффициентов
        self._C = C
        # матрица вероятностей
        self._Q = Q
        # пороговая низшая граница на начальном шаге
        self._min_freq = minimal_freq
    
    """
    Считаем кол-во частот в сети
    """
    def _count_units(self, matr):
        rows, cols = len(matr), len(matr[0])
        return len([matr[i][j] for i in range(rows) for j in range(cols) if matr[i][j] == 1])
    
    """
    Суммарное потребление линией группируем по каждой линии отдельно
    """
    def _group_by_total_freq_in_line(self, matr, lines):
        rows, cols = len(matr), len(matr[0])
        group_by_line = []
        for i in range(rows):
            # если линия занята пропускаем ее
            if (i in lines):
                continue
            # иначе добавляем пару значений: номер линии и сумма установленных частот
            group_by_line.append((i, sum([self._C[i][j] for j in range(cols) if matr[i][j] == 1])))    
        return group_by_line
    
    """
    Последовательность из более чем 1 единички в столбце
    """
    def _find_seq_of_units(self, matr):
        indexes = [] 
        matr_copy = copy.deepcopy(matr)
        m = matr_copy.transpose()    
        row, col = len(m), len(m[0])
        for i in range(row):
            units = [(j, i) for j in range(col) if m[i][j] == 1]        
            if len(units) > 1:
                indexes += units
        return indexes
    
    def _find_opt_line(self, matr, line):
        pass
    
    """
    Если вдруг опустимся ниже предельной низкой границы
    Нужно перераспределить частоты на некоторых линиях
    """
    def _redistribute_frequencys(self, matr, line):
        #print('-------redistribute--------')
        #print(matr)
        m = copy.deepcopy(matr)
        C_copy = copy.deepcopy(self._C)
        length = len(m[line])
        # занятые линии
        lines = [line]
        #operator.itemgetter()
        while sum([self._C[line][j] for j in range(length) if m[line][j] == 1]) < self._min_freq:
            # находим индекс максимальной частоты в линии
            j = C_copy[line].argmax()
            # если частота в линии занята
            if m[line][j] == 1:
                # обнуляем частоту
                C_copy[line][j] = 0
            else:
                # иначе устанавливаем и занимаем
                m[line][j] = 1
        rows, cols = len(m), len(m[0])
        # находим позиции единичек в матрице
        indexes = [(i, j) for i in range(rows) for j in range(cols) if m[i][j] == 1]
        # группируем сумму параметров частот по линиям
        group_by_line = self._group_by_total_freq_in_line(m, lines)
        # находим вторую по весу частот минимальную линию   
        next_line, _ = min(group_by_line, key=lambda p: p[1])
        # добавляем к занятым
        lines.append(next_line)
        while len([m[i][j] for i in range(rows) for j in range(cols) if m[i][j] == 1]) > len(m[0]):
            # находим индекс минимальной частоты в линии
            j = C_copy[next_line].argmin()
            # делаем ее максимально возможной
            C_copy[next_line][j] = sys.maxsize     
            m[next_line][j] = 0                       
            if sum([C_copy[next_line][j] for j in range(length) if m[next_line][j] == 1]) < self._min_freq:    
                # ищем новую линию
                next_line = min([(i, self._C[i][j]) for i, j in indexes if i not in lines], key=lambda p: p[1])
                # добавляем к занятым
                lines.append(next_line)
        # возвращаем перераспределенную матрицу
        #print(m)
        return m
    
    def _create_tasks(self, matr):
        tasks = []      
        indexes = self._find_seq_of_units(matr)
        for i, j in indexes:            
            matr_copy = copy.deepcopy(matr)
            changes = self._make_perm(matr_copy, i, j)
            #euristic relations
            ev_rel = sum([self._C[i][_j] for _j in range(len(changes[i])) if changes[i][_j] == 1]) 
            # если опустились ниже критической границы
            if ev_rel < self._min_freq:
                # нужно перерсапределить частоты
                changes = self._redistribute_frequencys(changes, i)
            tasks.append((ev_rel, changes))
        #print('длина {}'.format(len(tasks)))
        return tasks
    
    def _make_perm(self, matr, i, j):
        C_copy = copy.deepcopy(self._C)
        _j = C_copy[i].argmax()
        # начальные частоты повторно не должны использоваться
        while matr[i][_j] != 0 or (i, _j) in self._indexes: 
        #while matr[i][_j] != 0:            
            C_copy[i][_j] = 0              
            _j = C_copy[i].argmax()
        #clear value
        matr[i][j] = 0 
        #set new value
        matr[i][_j] = 1
        return matr
    
    def execute(self):
        counter = 0
        # дерево задач
        tree = []
        # получаем начальный список задач
        tasks = self._create_tasks(self._identity)
        # если оптимизировать нечго вернем полученные исх.данные
        if len(tasks) == 0:
            return (self._identity, self._ev_rel)
        # добавление в дерево решений
        tree.append(tasks)
        while True:
            print('текущий шаг: {}'.format(counter))   
            #print(tree)
            #print('-------------------------')
            # ищем задачу с максимальным весом качества связи в обрабатываемой линии
            max_ev_rel, maximize = max(tree[-1], key=lambda p: p[0])  
            # получаем подзадачи задачи с максимальным весом
            max_tasks = self._create_tasks(maximize)
            # если подзадачи пусты имеем оптимум
            if is_empty(max_tasks):
                return (max_ev_rel, maximize)
            # добавляем их в дерево задач            
            tree.append(max_tasks)
            # ищем максимальный вес подзадачи среди подзадач задачи с максимальным весом
            max_ev_rel_new, max_sub_task = max(max_tasks, key=lambda p: p[0])                 
            # оставляем на предпоследнем слое те задачи, вес которых больше веса максимальной подзадачи последнего левла
            tree[-2] = list(filter(lambda p: p[0] > max_ev_rel_new, tree[-2]))
            # если ничего не добавилось имеем оптимум
            if is_empty(tree[-2]):
                return (max_ev_rel_new, max_sub_task)
            # добавляем подзадачи из предпоследнего левла
            for _, curr_task in tree[-2]:                  
                task_to_append = self._create_tasks(curr_task)
                tree[-1] += task_to_append
            # удаляем все уровни кроме последнего 
            # т.к смысла дальше их хранить нет
            tail = tree[-1]
            # очищаем дерево
            tree.clear()    
            # добавляем первый левл
            tree.append(tail)  
            # увеличиваем уровень вложенности
            counter += 1
            
    def pad_matrix(matrix, pad_value=0):
        rows, cols = len(matrix), len(matrix[0])
        if rows == cols:
            return matrix
        
        new_matrix = []
        for r in matrix:
            tmp = []
            for c in r:
                tmp.append(c)
            new_matrix.append(tmp)
        
        if (rows > cols):
            for i in range(rows):
                new_matrix[i] += [pad_value] * (rows - cols)                
        else:
            while cols != len(new_matrix):
                new_matrix += [[pad_value] * cols]
        return new_matrix
                
class OptRelWSN:

    _Q = np.array([
    	[0.32, 0.83, 0.25, 0.26, 0.72],
    	[0.45, 0.53, 0.41, 0.47, 0.11],
    	[0.08, 0.14, 0.18, 0.63, 0.32]
    ])    
    
    # параметр уровня шума в канале связи
    noice = 5.0
    
    '''
    r - число строк
    c - число столбцов
    m - начальная матрица с вероятностями связи в группе каналов
    (ее заменяет _Q)
    '''
    def __init__(self, r=3, c=5, m=[]):
        self.n = r
        self.m = c
        self.Q = self._Q
        self.C = self._init_opt()
        
    def _logonormal_dist(self):
        scale = self.r * self.c
        mu, sigma = 0, 0.1 # mean and standard deviation
        lognorm_dist = np.random.lognormal(mean=0, sigma=self.noise, size=scale)
        matr = np.array(lognorm_dist.reshape(self.r, self.c))
        print(matr)
        return matr 
        
    """программа оптимального закрепления каналов связи за сенсорными узлами"""
    def _init_rel_wsn(self):        
        rnd = lambda x: np.round(np.random.uniform(0.035, 0.745), 2)
        a = list(map(rnd, np.arange(self.n * self.m)))
        matr = np.array(a).reshape(self.n, self.m)
        return matr
    
    def _init_opt(self):
        ln = lambda x: np.round(abs(np.log(1 - x)), 2)
        matr = copy.deepcopy(self.Q)
        return np.array(list(map(ln, matr)))
    
    def _make_identity(self, indexes):
        n, s = self.n, self.m
        identity = np.zeros(n * s, dtype=int).reshape(n, s)
        for i, j in indexes:
            identity[i][j] = 1
        return identity
    
    def _quality_param_in_line(self, line):
        # [self.Q[line][j] for j in range(self.n)]
        return 1 - reduce(operator.mul, self.Q[line, :])
    
    def compute(self):
        print(f'Исходная матрица:\n {self.C}')
        """
        @offsets - индексы частот, непопавшие в распределение
        @indexes - индексы частот, попавших в распределение
        """
        indexes, offsets = [], []
        C_copy = copy.deepcopy(self.C)
        for i in range(self.n):            
            j = C_copy[i].argmax()            
            while j in [_j for _, _j in indexes]:
                offsets.append((i, j))
                C_copy[i][j] = 0
                j = C_copy[i].argmax()
            indexes.append((i, j))
        """Исходная матрица для задач"""
        # в нее добавляем все смещения
        raw_task = indexes + offsets
        raw_identity = self._make_identity(raw_task)
        print(raw_identity)
        """ 
        Поиск наихудшей линии
        """
        frequencys = [(i, self.C[i][j]) for i, j in indexes]
        line, low_cost_line = min([(i, self.C[i][j]) for i, j in indexes], key=lambda x: x[1])
        """
        Добавляем частоты        
        """
        for j in range(len(self.C[line])):
            if (j not in [_j for _, _j in indexes]):
                indexes.append((line, j))                           
        """indexes += [(line, j) for j in range(len(self.C[line])) if j not in [_j for _, _j in indexes]]"""
        identity = self._make_identity(indexes)
        print(f'Эвристическая матрица:\n {identity}\n\n')
        frequencys = [(i, self.C[i][j]) for i, j in indexes]
        """
        min_freq = sys.maxsize
        for k, val in itertools.groupby(frequencys, lambda x: x[0]):
            pairs = list(val)
            print(pairs)
            total = sum([j for _, j in pairs])
            if total < min_freq:
                min_freq = total        
        """
        min_freq = min(list(map(lambda k: sum(list(map(lambda l: l[1], k[1]))), itertools.groupby(frequencys, lambda x: x[0]))))
        print(f'Нижняя эвристическая граница: {min_freq}\n')
        distribution = [(i, j) for i, j in indexes if j in [_j for _, _j in offsets]]
        """
        Для выбора оптимального набора частот
        """
        offsets += distribution
        min_rel_qual = 1 - reduce(operator.mul, self.Q[line])        
        min_total_rel = sum([self.C[line][i] for i in range(self.m) if raw_identity[line][i] == 1])        
        print(f'Нижняя граница: {min_total_rel}\n')
        """
        Создаем дерево задач
        """
        task = Task(self.C, self.Q, raw_identity, raw_task, min_total_rel, min_freq)
        """Выполняем разбор дерева"""
        res_tree_exec, matr_exec = task.execute()        
        min_rel_qual = 1 - reduce(operator.mul, [self.Q[line][j] for j in range(self.m) if matr_exec[line][j] == 1])
        print(f'Гарантированная надежность связи: {min_rel_qual}\n')
        print(f'Оптимальное распределение частот: {res_tree_exec}\n{matr_exec}')

def main():
    opt = OptRelWSN()
    opt.compute()

if __name__ == '__main__':
    main()
