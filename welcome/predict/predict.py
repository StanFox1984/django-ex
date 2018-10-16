#!/usr/bin/env python

import time
import sys
import random
import copy
import math
import os
import subprocess
import difflib
from threading import Thread
from multiprocessing.managers import BaseManager
from multiprocessing import Pool
from tinythreadpool import TinyThreadPool
from io import StringIO
from datetime import datetime
import sys

#from theano import *
#import theano.tensor as T
#from theano import function
#import numpy


class ProbTree:
    def __init__(self, parent):
        self.outcomes = {}
        self.pass_by_hits = { }
        self.endpoint_hits = { }
        self.data = None
        self.me = None
        self.probs = { }
        self.r1 = { }
        self.r2 = { }
        self.first = False
        self.stamp = time.time()
        self.total_hits = 0
        self.parent = parent
    def _addOutcome_Internal(self, key, data, last = False):
        p = None
        if key not in self.outcomes:
            if key not in self.parent.probnodes:
                p = ProbTree(self.parent)
                p.data = data
                p.me = ( key, data)
                self.parent.probnodes[key] = p
            else:
                p = self.parent.probnodes[key]
            self.outcomes[key] = p
            self.pass_by_hits[key] = 0
            self.endpoint_hits[key] = 0
        else:
            p = self.outcomes[key]
        if last == False:
            self.pass_by_hits[key] += 1
        else:
            self.endpoint_hits[key] += 1
        self.total_hits += 1
        return p
    def _recalcProbs(self):
        last_prob = 0.0
        for p in self.outcomes:
          self.probs[p] = float(self.pass_by_hits[p] + self.endpoint_hits[p])/float(self.total_hits )
          self.r1[p] = last_prob
          self.r2[p] = last_prob + self.probs[p]
          last_prob = self.r2[p]
    def recalcProbs(self, req_stamp = None):
        if req_stamp != None:
            if self.stamp == req_stamp:
                return
            self.stamp = req_stamp
        else:
            self.stamp = time.time()
        self._recalcProbs()
        for p in self.outcomes:
          self.outcomes[p].recalcProbs(self.stamp)
    def addOutcome(self, in_arr):
        tmp = self
        i = 0
        for el in in_arr:
          if i == len(in_arr)-1:
            last = True
          else:
            last = False
          tmp = tmp._addOutcome_Internal(el[0], el[1], last)
          i += 1
        self.recalcProbs()
    def printOutcomes(self, key = None, req_stamp = None):
        if req_stamp != None:
            if self.stamp == req_stamp:
                return
            self.stamp = req_stamp
        else:
            self.stamp = time.time()
        print (self.me, "Pass by hits: ", self.pass_by_hits," Endpoint hits: ",self.endpoint_hits," Key:", key, " Data:", self.data)
        print ("Probs:")
        for el in self.outcomes:
          print (self.probs[el], self.outcomes[el].me)

        for el in self.outcomes:
          self.outcomes[el].printOutcomes(el, self.stamp)
        print ("*************************")

    def getMostProbable(self):
        max_id = -1
        for p in range(0, len(self.outcomes)):
          if max_id != -1:
            if self.probs[self.outcomes[p]] > self.probs[self.outcomes[max_id]]:
              max_id = p
          else:
              max_id = p
        return self.outcomes[max_id]

    def generateWithProb(self, in_prefix, max_len, fuzzy = False):
        ret = [ ]
        tmp = self
        i = 0
        while True:
          if i >= len(in_prefix):
            break
          if in_prefix[i] not in tmp.outcomes:
            break
          tmp = tmp.outcomes[in_prefix[i]]
          i += 1
        if len(in_prefix) != i and fuzzy == False:
            return None
        while True:
          r = random.random()
          if len(tmp.outcomes) == 0:
              break
          for el in tmp.outcomes:
            if r < tmp.r2[el] and r >= tmp.r1[el]:
                ret.append((el, tmp.outcomes[el].data))
                if len(ret) > max_len:
                    return ret
                if len(tmp.outcomes[el].outcomes) > 0:
                    tmp = tmp.outcomes[el]
                    break
                else:
                    return ret
        return ret

class ProbNetwork:
    def __init__(self):
        self.probnodes = {}
    def addAssociativeChain(self, chain):
        vec = [ ( el.key, el ) for el in chain ]
        self.addOutcome(vec)
    def addOutcome(self, in_arr):
        p = None
        if in_arr[0][0] in self.probnodes:
            p = self.probnodes[in_arr[0][0]]
        else:
            p = ProbTree(self)
            p.me = in_arr[0]
            p.data = in_arr[0][1]
            p.first = True
            self.probnodes[in_arr[0][0]] = p
        p.addOutcome(in_arr[1:])
    def printOutcomes(self, key = None):
        if key == None:
            for el in self.probnodes:
                self.probnodes[el].printOutcomes(el)
        else:
            if key in self.probnodes:
                self.probnodes[key].printOutcomes()
    def getMostProbable(self, endpoint = False):
        max_tree = None
        for p in self.probnodes:
#          print ( self.probnodes[p].parent.data )
          if max_tree != None:
            if (self.probnodes[p].total_hits > max_tree.total_hits and self.probnodes[p].first == True) \
               or (max_tree.first == False and self.probnodes[p].first == True):
              max_tree = self.probnodes[p]
          else:
              max_tree = self.probnodes[p]
        return max_tree

    def generateWithProb(self, in_prefix, max_len, fuzzy = False):
        vec = [ el.key for el in in_prefix ]
        res = self._generateWithProb(vec, max_len, fuzzy)
        out = [ self.probnodes[el[0]].data for el in res ]
        return out

    def _generateWithProb(self, in_prefix, max_len, fuzzy = False):
        if len(in_prefix) > 0:
          if in_prefix[0] in self.probnodes:
            return self.probnodes[in_prefix[0]].generateWithProb(in_prefix[1:], max_len, fuzzy)
        else:
          return self.getMostProbable().generateWithProb(in_prefix[1:], max_len, fuzzy)

class Entity:
    def __init__(self, key, data):
        self.key = key
        self.data = data
    def __str__(self):
        return str(self.key) + "" +  str(self.data)
    @staticmethod
    def generateEntity(key):
        return Entity(key, "Entity" + str(key))

def getVecDiffAbs(str1, str2):
    n = min([len(str1), len(str2)])
    res = 0
    for i in range(0, n):
      ch1 = str1[i]
      ch2 = str2[i]
      res += abs(int(ch1)-int(ch2))
    return res

def getVecDiff(str1, str2):
    n = min([len(str1), len(str2)])
    res = 0
    for i in range(0, n):
      ch1 = str1[i]
      ch2 = str2[i]
      res += int(ch1)-int(ch2)
    return res

def getVecAbs(vec):
    n = 0
    for v in vec:
      n += abs(v)
    return n

def getArrCorrelations( X, Y):
    prev_X = None
    prev_Y = None
    Corr = [ ]
    for j in range(0, len(X[0])):
      Corr.append(0)
    for i in range(0, len(X)):
      if prev_X != None:
        for j in range(0, len(X[i])):
          if prev_X[j] != X[i][j]:
            res = getVecDiffAbs(prev_Y, Y[i])/getVecDiffAbs(prev_X, X[i])
            if res != 0:
              Corr[j] += res
            else:
              Corr[j] += 1
      prev_X = X[i]
      prev_Y = Y[i]
    return Corr

def getVecDiffAbsWithCorr(str1, str2, corr):
    norm = max(corr)
    _corr = [ float(c)/float(norm) for c in corr ]
    n = min([len(str1), len(str2)])
    res = 0
    for i in range(0, n):
      ch1 = str1[i]
      ch2 = str2[i]
      res += _corr[i]*(abs(int(ch1)-int(ch2)))
    return res

class BitArray:
    def __init__(self, initial_bits = 32, int_size = 32):
        self.array = [ 0 for i in range(0, (initial_bits / int_size) ) ]
        self.int_size = int_size
    def getBit(self, bit):
        octet = bit / self.int_size
        offset = bit % self.int_size
        if octet > len(self.array):
            self.array.extend( [ 0 for i in range(octet  - len(self.array)) ] )
        n = 1 << offset
        if self.array[octet] & n:
            return 1
        return 0
    def getNumber(self, _offset, num_bits = None):
        if num_bits == None:
            num_bits = self.int_size
        octet = _offset / self.int_size
        offset = _offset % self.int_size
        if octet > len(self.array):
            self.array.extend( [ 0 for i in range(octet  - len(self.array)) ] )
        shift = 0
        number = 0
        n = 1 << offset
        while shift < num_bits:
            if n & self.array[octet]:
                number |= n
            n = n << 1
            shift += 1
        number = number >> offset
        return number
    def setBit(self, bit, value):
        octet = bit / self.int_size
        offset = bit % self.int_size
        if octet > len(self.array):
            self.array.extend( [ 0 for i in range(octet  - len(self.array)) ] )
        shift = 0
        n = value
        while shift != offset:
            n = n << 1
            shift += 1
        if n == 1<<shift:
          self.array[octet] |= n
        if n == 0:
          self.array[octet] &= ~n
    def setArray(self, arr):
        self.array = [ ]
        for a in arr:
          b = BitArray()
          b.setNumber(0, int(a))
          self.extendWith(b)
    def setNumber(self, _offset, value, bits_in_value = None):
        if bits_in_value == None:
            bits_in_value = self.int_size
        octet = _offset / self.int_size
        offset = _offset % self.int_size
        if octet > len(self.array):
            self.array.extend( [ 0 for i in range(octet  - len(self.array)) ] )
        shift = 0
        n = value
        while shift != offset:
            n = n << 1
            shift += 1
        i = 1<<shift
        bits = 0
        while bits < bits_in_value:
            if i & n:
                self.array[octet] |= i
            else:
                self.array[octet] &= ~i
            i = i<<1
            bits += 1
    def extendWith(self, bit_array):
        self.array.extend(bit_array.array)
    def copyFrom(self, bit_array):
        self.array = [ ]
        self.array.extend(bit_array.array)
        self.int_size = bit_array.int_size
    def printBits(self, normal = False):
        for i in reversed(self.array):
          if normal == False:
            shift = pow(2, self.int_size)
            while shift != 0:
                if shift & i:
                    sys.stdout.write("1")
                else:
                    sys.stdout.write("0")
                sys.stdout.flush()
                shift = shift>>1
          if normal == True:
            sys.stdout.write(str(i))
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
    def getAsOne(self):
        s = ""
        for i in reversed(self.array):
            s += str(i)
        return int(s)

class Factor:
    def __init__(self, offset, width, input_function):
        self.offset = offset
        self.width = width
        self.input_function = input_function
    def invokeInput(self):
        res = self.input_function()
        return res

class FactorManager:
    def __init__(self):
        self.factors = [ ]
        self.bit_array = BitArray()
        self.last_offset = 0
    def addFactor(self, width, input_function):
        self.factors.append(Factor(self.last_offset, width, input_function))
        self.last_offset += width
    def refillState(self):
        for factor in self.factors:
            self.bit_array.setNumber(factor.offset, factor.invokeInput(), factor.width)
        return self.bit_array.getAsOne()


def _distance_func(factor1, factor2):
    seq=difflib.SequenceMatcher(a=str(factor1).lower(), b=str(factor2).lower())
    r = int(seq.ratio()*100.0)
    return r

def prefix_match_len(str1, str2):
    l = min([len(str1), len(str2)])
    n = 0
    for i in range(0, l):
      if str1[i] == str2[i]:
        n+=1
      else:
        break
    return n

def generate_entity_x(factor):
    return Entity(factor, factor)

class FactorAnalyzer:
    def __init__(self):
        self.factor_map = { }
        self.prob_network = ProbNetwork()
        self.last_state = None
        self.total_factor_hits = 0
        self.corr = [ ]
        self.factor_outcome_list = [ ]
    def Print(self):
        self.prob_network.printOutcomes()
    def analyze_factor(self, factor, generate_func = None, outcome = None):
        state = None
        if factor not in self.factor_map:
            if generate_func == None:
              state = Entity.generateEntity(factor)
            else:
              state = generate_func(factor)
            self.factor_map[factor] = state
        else:
            state = self.factor_map[factor]
        if self.last_state == None:
            self.last_state = state
        else:
            self.prob_network.addAssociativeChain([ self.last_state, state])
            self.last_state = state
        if outcome != None:
          self.factor_outcome_list.append((factor, outcome))
          self.corr = getArrCorrelations([ eval(r[0]) for r in self.factor_outcome_list ], [ eval(r[1]) for r in self.factor_outcome_list ])
    def deduct(self, prefix_factors, depth, generate_func = None, distance_func = _distance_func):
        distance_metric = 0
        closest_factor  = None
        prefix_states = [ ]
        m0 = 0
        i = 0
        fuzzy = False
        for factor in prefix_factors:
            if factor not in self.factor_map:
                fuzzy = True
                for el in self.factor_map:
                    if len(self.factor_outcome_list) > 0:
                      d = getVecDiffAbsWithCorr(eval(el), eval(factor), self.corr)
                      m = prefix_match_len(el, factor)
                    else:
                      if distance_func != None:
                        d = distance_func(el, factor)
                      else:
                        return None
                    if closest_factor == None or d < distance_metric:
                        closest_factor = el
                        distance_metric = d
                        m0 = m
                    elif d == distance_metric and m > m0:
                        closest_factor = el
                        distance_metric = d
                        m0 = m
            else:
                closest_factor = factor
            print ( "closest was", closest_factor )
            prefix_factors[i] = closest_factor
            i += 1
            if generate_func == None:
              prefix_states.append(Entity.generateEntity(closest_factor))
            else:
              prefix_states.append(generate_func(closest_factor))
        if depth <= len(prefix_states):
          return prefix_states
        res =  self.prob_network.generateWithProb(prefix_states, depth-len(prefix_states), fuzzy)
        prefix_states.extend(res)
        return prefix_states

# Gradient calculation for backpropagation network:
#Yj = sum (Xij Wj)
#Err func(Wj) = sum_m( Ymj - sum_i(XmiWj) )^2    j = 0..len(X), 0..len(Y), i = range(j), m= 0..len(observations)
#gradient(Err_func(Wj), j) = sum_m ( -2 * Ymj * sum_i(Xmi) + sum_i(Xmi*2*Wj))

#Err func(Wj) = ( Yj - sum_i(XiWj) )^2    j = 0..len(X), 0..len(Y), i = range(j), m= 0..len(observations)

#Err func2(Wj) = sum_m( Ymj - sum_i(XmiWj) )^2    j = 0..len(X), 0..len(Y), i = range(j), m= 0..len(observations)
#gradient(Err_func2(Wj), j) =  -1 * sum_m(-2 * Ymj * sum_i(Xi) + sum_i(Xmi*2*Wj))

#d(Ymj^2 - 2 Ymj sum_i(XmiWj) + (sum_i(XmiWj)^2)/dWj = sum_m(-2 Ymj sum_i(Xmi) + sum_i(Xmi^2*2*Wj))

def array_sum(arr):
    r = sum(arr)
#    print ( "array_sum res:",r, "args:", arr )
    return r

def array_sum_theano(arr):
    v = T.vector('v')
    z = T.sum(v)
    f = function([ v ], z)
    res = f(arr)
    return res

def array_sum_i(arr, i):
    return sum(arr[i])

def array_sum_i_theano(arr, i):
    return array_sum_theano(arr[i])

def multiply_scalars(a1, a2):
    r = a1*a2
#    print ( "multiply_scalars res:", r, "args:", a1, a2 )
    return r

def multiply_scalars_theano(a1, a2):
    x = T.dscalar('x')
    y = T.dscalar('y')
    z = x*y
    f = function( [ x,y],z)
    res = f(a1,a2)
    return res

def multiply_scalar_array(a1, arr1):
    r = [ a1*a for a in arr1 ]
#    print ( "multiply_scalar_array res:", r, "args:", a1, arr1 )
    return r

def multiply_scalar_array_theano(a1, arr1):
    x = T.dscalar('x')
    v = T.vector('v')
    z = v*x
    f = function( [x, v], z)
    res = f(a1, arr1)
    return res

def array_sum_multiply(arr1, arr2):
    return array_sum([ arr1[i]*arr2[i] for i in range(0, len(arr1)) ])

def array_sum_multiply_theano(arr1, arr2):
    v1 = T.vector('v1')
    v2 = T.vector('v2')
    z = T.sum(v1*v2)
    f = function([v1,v2], z)
    res = f(arr1,arr2)
    return res

def array_sum_multiply_i(arr1, arr2, i):
    r = array_sum([ arr1[i][j]*arr2[i][j] for j in range(0, len(arr1[i])) ])
#    print ( "array_sum_multiply_i res:", r, "args: ", arr1, arr2, i )
    return r

def array_sum_multiply_i_theano(arr1, arr2, i):
    return array_sum_multiply_theano(arr1[i], arr2[i])

def gradient(Y, X, W, grad):
    for j in range(0, len(W)):
      grad[j] = 0
      for m in range(0, len(Y)):
        g = multiply_scalars(2.0*Y[m][j], array_sum(X[m])) + array_sum(multiply_scalar_array(-2.0*W[j], [ X[m][i]*X[m][i] for i in range(0, len(X[m])) ] ))
#        print ( "gradient calc res: ", g )
        grad[j] += g

def gradient_theano(Y, X, W, grad):
    for j in range(0, len(W)):
      for m in range(0, len(Y)):
        grad[j] += multiply_scalars_theano(2.0, array_sum_multiply_i_theano(X, Y, m)) + array_sum_theano(multiply_scalar_array_theano(-2.0*W[j], [ X[m][i]*X[m][i] for i in range(0, len(X[m])) ] ))

class NeuralLinearLayer:
    def __init__(self, W0, step = None, max_iterations = 10):
      self.W = list(W0)
      self.last_err_func = None
      if step:
        self.step = list(step)
      else:
        self.step = [ 0.001 for n in self.W ]
      self.max_iterations = max_iterations
      self.WC = [ 0.0 for i in range(0,len(self.W)) ]
      self.step_multiplier = 2
      self.step_multiplier_local_opt = 8
      self.err = 0.0000001
      self.delay_tolerance = 200000

    def set_multipliers(self, step_multiplier, step_multiplier_local_opt):
      self.step_multiplier = step_multiplier
      self.step_multiplier_local_opt = step_multiplier_local_opt

    def set_err(self, err):
      self.err = err

    def set_delay_tolerance(self, delay):
      self.delay_tolerance = delay

    def study2(self, X, Y):
      grad = [ 0 for n in self.W ]
      Wout = list(self.W)
      iterations = 0
      overall_iterations = 0
      self.last_err_func = None
      self.n = None
      grad_zero = True
      n = 0
      iterations_left = self.max_iterations
      last_grad = None
      first = True
      old_t = datetime.now()
      while True and iterations < iterations_left:
        if iterations >= (iterations_left/4) or ((datetime.now() - old_t).microseconds>self.delay_tolerance):
          old_t = datetime.now()
          for l in range(0, len(self.step)):
            if grad_zero != True:
              self.step[l] *= self.step_multiplier
            else:
              self.step[l] *= self.step_multiplier_local_opt
#            print ( "step_increased ", self.step, "error: ", n, "W:", self.W, "grad:", grad, last_grad )
          overall_iterations += iterations
          iterations_left -= iterations
          for h in range(0, len(X)):
            _Y = copy.deepcopy(Y)
            t = self.calc_y(X[h], _Y[h])
            _Y[h] = copy.deepcopy(t)
          iterations = 0
        g = self.calc_gradient2(X, Y, grad)
#        grad = copy.deepcopy(g)
        for j in range(0, len(self.W)):
#          if last_grad[j] !=0 and last_grad[j]!=grad[j]:
#            for l in range(0, len(self.step)):
#              self.step[l] /= self.step_multiplier
#            print ( "Gradient change detected, decreasing step to", self.step, n, last_grad, grad )
          Wout[j] = self.W[j]
          self.W[j] += self.step[j]*grad[j]
        n = self.calc_err_func2(X, Y)
        grad_zero = True
        for h in range(0, len(grad)):
          if int(grad[h]) != 0:
            grad_zero = False
            break
        if grad_zero == True:
          if n <= self.err:
            break
          else:
            for j in range(0, len(self.W)):
              Wout[j] = self.W[j]
              if last_grad != None:
                self.W[j] += self.step[j]*(last_grad[j])
              else:
                self.W[j] += self.step[j]
              n = self.calc_err_func2(X, Y)
#            print ( "step_increased opt", self.step, n, grad, self.W, last_grad )
        else:
          last_grad = copy.deepcopy(grad)
        if (self.last_err_func == None or n <= self.last_err_func) and n >= self.err:
          self.last_err_func = n
        else:
          for j in range(0,len(self.W)):
            self.W[j] = Wout[j]
          n = self.calc_err_func2(X, Y)
          Ytest = copy.deepcopy(Y)
          break
        iterations += 1
      overall_iterations += iterations
#      print ( "Ended up with W: ", self.W )

    def calc_y(self, X, Y):
        for j in range(0, len(Y)):
          Y[j] = 0.0
          for i in range(0, len(X)):
            if abs(X[i] - 0.0) < 0.0000001:
              X[i]+=0.0000001
            Y[j]+=(self.W[j])*X[i]
          Y[j]+=self.WC[j]
        return Y

    def calc_gradient2(self, X, Y, grad):

      gradient(Y, X, self.W, grad)

      for j in range(0, len(self.W)):
        if grad[j] != 0:
          grad[j] = grad[j]/abs(grad[j])
      return grad

    def calc_err_func2(self, X, Y):
      Err = 0
      for m in range(0, len(Y)):
        for j in range(0, len(Y[m])):
          s = 0.0
          for i in range(0, len(X[m])):
            s += X[m][i]*self.W[j]
#          s+=self.WC[j]

          Err += pow(( Y[m][j] - s ), 2)
      return Err




class NeuralLinearNetwork:
    def __init__(self, W0, num_layers, step = None, max_iterations = 10, sp = 1):
      self.layers = [ ]
      self.stop_layer = None
      W = W0
      for i in range(0, num_layers):
          self.layers.append(NeuralLinearLayer(W, step, max_iterations))
      self.sp = sp
      self.last_yout = None

    def set_multipliers(self, step_multiplier, step_multiplier_local_opt):
      for i in range(0, len(self.layers)):
          self.layers[i].set_multipliers(step_multiplier, step_multiplier_local_opt)

    def study(self, X, Y):
      Y0 = copy.deepcopy(Y)
      X0 = copy.deepcopy(X)
      X_layers = [ ]
      Out = copy.deepcopy(Y)
      for m in range(0, len(Out)):
        for j in range(0, len(Out[m])):
          Out[m][j] = 0
      for i in range(0, len(self.layers)):

        self.layers[i].study2(X0, Y0)
        X_layers.append(X0)

        for m in range(0, len(Y0)):
          YY = self.layers[i].calc_y(X0[m], Y0[m])
          for j in range(0, len(Y0[m])):
            Out[m][j] += Y0[m][j]

        X0 = copy.deepcopy(Out)
        for m in range(0, len(Y0)):
          for j in range(0, len(Y0[m])):
            Y0[m][j] = Y[m][j] - Out[m][j]

      self.stop_layer = None
      Err = [ ]
      Y0 = [ 0 for i in range(0, len(Y[0])) ]
      for i in range(0, len(self.layers)):
        s = 0
        for m in range(0, len(Y)):
          YY = self.calc_y2(X[m], Y0, i + 1)
          for y in range(0, len(Y0)):
            Y0[y] = copy.deepcopy(YY[y])
          for j in range(0, len(Y[m])):
            s += abs(Y0[j] - Y[m][j])
        Err.append(s)


      self.stop_layer = Err.index(min(Err)) + 1

    def getWeights(self, layer):
      return self.layers[layer].W

    def calc_y2(self, X, Y, up_to = None):
      X0 = list(X)
      Y0 = list(Y)
      Out = [ 0 for n in range(0, len(Y)) ]
      if self.stop_layer != None:
        max_layer = self.stop_layer if self.stop_layer < len(self.layers) else len(self.layers)
      else:
        max_layer = len(self.layers)
        if up_to != None:
          max_layer = up_to if up_to < len(self.layers) else len(self.layers)
      for i in range(0, max_layer):
        YY = self.layers[i].calc_y(X0, Y0)
        for y in range(0, len(Y0)):
          Y0[y] = copy.deepcopy(YY[y])
        for j in range(0, len(Out)):
          Out[j] += Y0[j]
        X0 = list(Out)
      for i in range(0, len(Out)):
        Y[i] = Out[i]
      return Y

class NeuralNetworkManager(BaseManager):
    pass

NeuralNetworkManager.register('Network', NeuralLinearNetwork)

def createNetworkForNode(node_addr, _authkey, W0, num_layers, step = None, max_iterations = 10):
    m = NeuralNetworkManager(address=node_addr, authkey=_authkey)
    m.connect()
    network = m.Network(W0, num_layers, step, max_iterations)
    return network


def autoCorrelation( Y, n ):
  Corr = 0
  for m in range(0, len(Y)-n):
    Corr+=abs((Y[m][0] - Y[(m+n) % len(Y)][0]))
  Corr = Corr / (len(Y) - n)
  return Corr

def detectMonotonicity(Y):
  if len(Y) <= 1:
    return False
  last_Y = Y[0][0]
  if Y[1][0] != last_Y:
    last_s = (Y[1][0] - last_Y) / (abs(Y[1][0] - last_Y))
  else:
    last_s = 0
  c = 0
  for i in range(2, len(Y)):
    if Y[i][0] != last_Y:
      s = (Y[i][0] - last_Y) / (abs(Y[i][0] - last_Y))
    else:
      s = 0
    if last_s != s and last_s != 0:
      c += 1
    last_Y = Y[i][0]
    last_s = s
  return c >= 1


def _detectPeriodic( Y ):
  prev_corr = None
  Corr = None
  min_corr = None 
  num_periods = 0
  df = 0
  prev_df = 0
  for i in range(0, len(Y)):
    prev_corr = Corr
    Corr = autoCorrelation( Y, i )
    if prev_corr != None:
      prev_df = df
      df = abs(prev_corr - Corr)
      print ( df )
      if prev_corr > Corr:
        num_periods += 1
        if min_corr == None or min_corr[0] > Corr:
          min_corr = ( Corr, i )
    else:
      prev_corr = Corr

  if min_corr != None and num_periods >= 1:
    return True
  return False

class NeuralLinearComposedNetwork:
    def __init__(self, points_per_network, W0, num_layers, step = None, max_iterations = 10, parallelize = False, sp = 1):
      self.networks = [ ]
      self.W0 = W0
      self.num_layers = num_layers
      self.step = step
      self.parallelize = parallelize
      self.max_iterations = max_iterations
      self.sp = sp
      self.points_per_network = points_per_network
      self.cyclic  = [ False for j in self.W0 ]
      self.mn = [ [ None ] for j in range(0, len(W0)) ]
      self.mx = [ [ None ] for j in range(0, len(W0)) ]
      self.pool = None
      self.acc_x = [ ]
      self.acc_y = [ ]
      self.step_multiplier = 2
      self.step_multiplier_local_opt = 8

    def autoCorrelation( self, Y, n ):
      return autoCorrelation(Y, n)

    def detectPeriodic( self, Y ):
      return detectMonotonicity(Y)

    def nstudy_wrapper(self, network, X, Y):
      network.study(X, Y)

    def set_multipliers(self, step_multiplier, step_multiplier_local_opt):
      self.step_multiplier = step_multiplier
      self.step_multiplier_local_opt = step_multiplier_local_opt

    def study(self, X, Y):
      self.acc_y.extend(Y)
      self.acc_x.extend(X)
      for j in range(0, len(self.W0)):
        self.cyclic[j] = self.detectPeriodic( [ [ self.acc_y[m][j] ] for m in range(0, len(self.acc_y)) ] )
        self.cyclic[j] = self.cyclic[j] or self.detectPeriodic( [ [ self.acc_x[m][j] ] for m in range(0, len(self.acc_x)) ] )
      n = 0
      added_X = [ ]
      added_Y = [ ]
      for m in range(0, len(X)):
        added_X.append( X[m] )
        added_Y.append( Y[m] )
        n+=1
        if n >= self.points_per_network or m == len(X)-1:
          n = 0
          _added_X = [ [ u for u in range(0, len(added_X))] for u1 in range(0, len(self.W0)) ]
          for j in range(0, len(self.W0)):
            for _m in range(0, len(added_X)):
              _added_X[j][_m]= added_X[_m][j]

          mn = [ min(x) for x in _added_X ]
          mx = [ max(x) for x in _added_X ]

          found = False
          ii = 0
          for network in self.networks:
            ii += 1
            if network[0] <= mn and network[1] >= mx:
              if self.parallelize == False:
                network[2].study(added_X, added_Y)
              else:
                if self.pool == None:
                  self.pool = TinyThreadPool(10)
                  self.pool.start()

                self.pool.enqueue_task_id(ii % 10, NeuralLinearComposedNetwork.nstudy_wrapper,self, network[2], added_X, added_Y)

              found = True
              break
          if found == False:
            if self.parallelize == False:
              self.networks.append( ( mn, mx, NeuralLinearNetwork( self.W0 , self.num_layers, self.step, self.max_iterations, self.sp) ) )
            else:
              self.networks.append( ( mn, mx, createNetworkForNode(('', 50000), 'abc', self.W0, self.num_layers, self.step, self.max_iterations) ) )
#            print ( "Added network ", mn, mx, len(self.networks)-1 )

            self.networks[len(self.networks)-1][2].set_multipliers(self.step_multiplier, self.step_multiplier_local_opt)
            if self.parallelize == False:
              self.networks[len(self.networks)-1][2].study(added_X, added_Y)
            else:
              if self.pool == None:
                self.pool = TinyThreadPool(10)
                self.pool.start()
              self.pool.enqueue_task_id((len(self.networks)-1)%10,NeuralLinearComposedNetwork.nstudy_wrapper,self, self.networks[len(self.networks)-1][2],added_X, added_Y)

            for j in range(0, len(self.W0)):
              assign = False
              if self.mn[j][0] == None:
                  assign = True
              else:
                  if mn[j] < self.mn[j][0]:
                      assign = True
              if assign:
                  self.mn[j] = ( mn[j], len(self.networks)-1 )
              assign = False
              if self.mx[j][0] == None:
                  assign = True
              else:
                  if mx[j] > self.mx[j][0]:
                      assign = True
              if assign:
                  self.mx[j] = ( mx[j], len(self.networks)-1 )

          added_X = [ ]
          added_Y = [ ]
          if self.pool != None:
            self.pool.wait_ready()

    def getWeights(self, mn, mx, j, layer):
      for network in self.networks:
        if network[0] <= mn and network[1] >= mx:
          return network[2].getWeights(layer)

    def calc_y2(self, X, Y, up_to = None):
      network = None
      i = None
      num_match = [ 0 for j in self.networks ]
      distance = [ [ 0, 0 ] for j in self.networks ]
      found = False
      XX = copy.deepcopy(X)
      for j in range(0, len(self.W0)):
        _X = [ ]
        if self.cyclic[j] == True:
#          XX[j] = self.mn[j][0] + X[j] % ( self.mx[j][0] - self.mn[j][0] + 1)
          XX[j] =  X[j] % ( self.mx[j][0]+1)
          _X.append(XX[j])
        else:
          _X.append(X[j])
        for n in range(0, len(self.networks)):
          if (self.networks[n][0][j] <= _X[0] and self.networks[n][1][j] >= _X[0]):
            network = self.networks[n]
            i = n
            distance[n][0] = abs(self.networks[n][0][j] - _X[0] )
            distance[n][1] = abs(self.networks[n][1][j] - _X[0] )
#            num_match[n] += 1
          else:
            distance[n][0] = abs(self.networks[n][0][j] - _X[0] )
            distance[n][1] = abs(self.networks[n][1][j] - _X[0] )
            i = n
          num_match[n] += 1 + (-0.5)*(distance[n][0]+distance[n][1])
      if i != None:
#        print ( "chosen network num ", num_match.index(max(num_match)), "for X ", XX, " of ", [ (self.networks[p], num_match[p]) for p in range(0,len(num_match)) ] )
        ns = [ (num_match[p],self.networks[p]) for p in range(0,len(num_match)) ]
        ns=sorted(ns,reverse=True)
#        print ( ns )
#c/a1 + c/a2 + c/a3 = 1

#1/a1 +1/a2 + 1/a3 = 1/c
#c = 1/(1/a1+1/a2+1/a3)

        ns2 = ns[:int(len(ns)/8)]
#        print ( ns2 )
        nm = sum([ 1/abs(t[0]) for t in ns2 ])

        if nm != 0:
          nm = 1/nm
          prob_array = [ ]
          last_prob = 0

          for u in range(0, len(ns2)):
            prob_array.append((last_prob, last_prob+nm/abs(ns2[u][0]),ns2[u][1][2],ns2[u]))
            last_prob += nm/abs(ns2[u][0])

          r =random.random()
#          print ( prob_array )
          for u in range(0, len(prob_array)):
            if r>=prob_array[u][0] and r<=prob_array[u][1]:
              YY = prob_array[u][2].calc_y2(XX, Y, up_to)
#              print ( "chosen network ",prob_array[u][2], "with probability ", prob_array[u][0], prob_array[u][1], r, prob_array[u][2] )
              break

#          YY = copy.deepcopy(Y)
#          for u in range(0, len(Y)):
#            YY[u] = 0
#            for net in ns2:
#              net[1][2].calc_y2(XX, Y, up_to)
#              print ( abs(nm/net[0]), Y[u], abs(nm/net[0])*Y[u], net, XX )
#              YY[u]+=abs(nm/net[0])*Y[u]
        else:
          YY = self.networks[num_match.index(max(num_match))][2].calc_y2(XX, Y, up_to)
#        YY = self.networks[num_match.index(max(num_match))][2].calc_y2(XX, Y, up_to)
        for y in  range(0, len(Y)):
          Y[y] = round(copy.deepcopy(YY[y]),2)
      else:
        dist = [ d[0] + d[1] for d in distance ]
#        print ( "chosen network num ", dist.index(min(dist)),"for X ", XX )
        YY = self.networks[dist.index(min(dist))][2].calc_y2(XX, Y, up_to)
        for y in  range(0, len(Y)):
          Y[y] = round(copy.deepcopy(YY[y]),2)
      return XX

def getMean(mvec):
    s = [ 0 for i in range(0, len(mvec[0])) ]
    for m in range(0, len(mvec)):
      for j in range(0, len(mvec[m])):
        s[j] += mvec[m][j]
    mid = [ float(s[i])/float(len(mvec)) for i in range(0, len(s)) ]
    return mid

def getVecDelta(vec1, vec2):
    vec = [ 0 for i in range(0, len(vec1)) ]
    for v in range(0, len(vec1)):
      vec[v] = abs(vec1[v]-vec2[v])
    return vec

def getAverageDelta(vec, mean):
    s = 0.0
    for v in vec:
      s+= getVecDiffAbs(v, mean)
    s = s / float(len(vec))
    return s

def getVecAverageDelta(vec, mean):
    s = [ 0.0 for i in range(0,len(vec[0])) ]
    for v in vec:
      d = getVecDelta(v, mean)
      for j in range(0, len(vec[0])):
        s[j] += d[j]
    for j in range(0, len(vec[0])):
      s[j] = s[j] / float(len(vec))
    return s

def getVecMaxDelta(vec, mean):
    max_delta = [ 0.0 for i in range(0,len(vec[0])) ]
    for v in vec:
      d = getVecDelta(v, mean)
      for j in range(0, len(vec[0])):
        if abs(max_delta[j]) < abs(d[j]):
          max_delta[j] = d[j]
    return max_delta

class Cluster(object):
    def __init__(self, vec = [ ], parent = None, name = None):
      self.vec = [ ]
      if len(vec) > 0:
        if type(vec[0]) != list:
          self.vec.append(vec)
        else:
          self.vec = copy.deepcopy(vec);
      self.mean = None
      self.av_delta = None
      self.err = 1
      self._recalc()
      self.parent = parent
      self.subclusters = [ ]
      self.name = name
    def _recalc(self):
      if len(self.vec) > 0:
        self.mean = getMean(self.vec)
        self.av_delta = getVecMaxDelta(self.vec, self.mean)
        self.err = abs(max(self.av_delta)/2)+self.err
        return True
      return False
    def check_delta(self, vec):
      d1 = getVecDelta(vec, self.mean)
      for j in range(0, len(d1)):
        self.err = abs(self.mean[j]/2)
        if (d1[j] - self.av_delta[j]) > self.err:
          return False
      return True
    def classify(self, vec, no_add = False):
      if not self._recalc():
        self.vec.append(vec)
        self._recalc()
        return True
      left = True
      if not self.check_delta(vec):
          left = False
      if left == True:
        if no_add != True:
          self.vec.append(vec)
          self._recalc()
        return True
      return False
    def clusterize(self):
        if not self._recalc():
          return [ ]
        vec1 = [ ]
        vec2 = [ ]
        for v in self.vec:
          left = True
          if not self.check_delta(v):
              left = False
          if left == True:
            vec1.append(v)
          else:
            vec2.append(v)
        if len(vec1)>0 and len(vec1)!=len(self.vec):
          c1 = Cluster(vec1, self)
          self.subclusters.append(c1)
        if len(vec2)>0 and len(vec2)!=len(self.vec):
          c2 = Cluster(vec2, self)
          self.subclusters.append(c2)
        return self.subclusters

    def k_means(cl, vec, num_splits=None):
        k = [ ]
        if num_splits == None:
          num_splits = len(vec)
        else:
          num_splits = min([len(vec), num_splits])
        clusters = cl.clusterize()
        if num_splits <= 0:
          return clusters
        for c in clusters:
          k.append(k_means(c, c.vec, num_splits-1))
        return k

    def k_means_old(cl, vec, num_splits=None):
        k = [ ]
        if num_splits == None:
          num_splits = len(vec)
        else:
          num_splits = min([len(vec), num_splits])
        c = Cluster()
        c.parent = cl
        for v in vec:
          classified = False
          for cc in k:
            if cc.classify(v):
              classified = True
          if classified != True and not c.classify(v):
            k.append(c)
            c = Cluster(v, cl)
        if len(c.vec) > 0:
          k.append(c)
        k1 = [ ]
        for c in k:
          if ((num_splits - 1) > 0) and ((len(c.vec))>1):
            r1 = Cluster.k_means(c, c.vec, num_splits-1)
            if len(r1) > 0:
              k1.extend(r1)
        k.extend(k1)
        return k
    def __lt__(self, other):
        return str(self.vec) < str(other.vec)
    def __str__(self):
        c = self
        s = "Cluster:"+ object.__str__(self) +" Parent: "+ str(c.parent) + " Vec: "+ str(c.vec) + " Mean: " + str(c.mean) + " Av delta: "+ str(c.av_delta) + "\n"
        s += "Subclusters: " + str(c.subclusters) + "\n"
        return s
    def print_info(self):
        print ( str(self) )

class Classificator:
    def __init__(self, init_vec = None):
      self.init_vec = [ ]
      self.clusters = [ ]
      self.cluster = None
      if init_vec != None:
        self.reinit(init_vec)
    def add_cluster(self, cluster):
      self.init_vec.extend(cluster.vec)
      if self.cluster == None:
        self.cluster = Cluster(self.init_vec)
      self.cluster.subclusters.append(cluster)
      self.clusters.append(cluster)
    def reinit(self, init_vec, add = True):
      if add == True:
        self.init_vec.extend(init_vec)
      else:
        self.init_vec = copy.deepcopy(init_vec)
      self.cluster = Cluster(self.init_vec)
      self.clusters  = Cluster.k_means_old(self.cluster, self.init_vec, 1)
      print ( self.clusters )
    def classify_vec(self, vec, first_or_smallest = False, only_first = True):
      cluster_map = { }
      for v in range(0, len(vec)):
        c = self.classify(vec[v], first_or_smallest)
        if c!=None:
          if not c in cluster_map:
            cluster_map[c] = 1
          else:
            cluster_map[c] += 1
        else:
            if not self.cluster in cluster_map:
              cluster_map[self.cluster] = 1
            else:
              cluster_map[self.cluster] += 1
      cluster_vec = [ (c,cluster_map[c]) for c in cluster_map ]
      clusters = [ ]
      for c in cluster_vec:
          if c[1] > 0:
            clusters.append(c)
      clusters.sort()
      if only_first == True:
        if len(clusters)>1:
          return clusters[1][0]
        else:
          return clusters[0][0]
      return clusters
    def classify(self, vec, first_or_smallest = False):
      smallest = None
      if first_or_smallest == True:
        for c in self.clusters:
          if c.classify(vec, True):
            return c
      else:
        for c in self.clusters:
          if c.classify(vec, True):
            if smallest != None:
              if len(c.vec) < len(smallest.vec):
                smallest = c
            else:
                smallest = c
        return smallest
      return None
    def print_info(self):
      for c in self.clusters:
        c.print_info()

class FuzzyDict(dict):
    """A dictionary that applies an arbitrary key-altering function
       before accessing the keys."""

    def __keytransform__(self, key):
        return key

    # Overridden methods. List from 
    # http://stackoverflow.com/questions/2390827/how-to-properly-subclass-dict

    def __init__(self, *args, **kwargs):
        self.ranges = [ ]
        self.err = 0.5
        self.update(*args, **kwargs)

    # Note: I'm using dict directly, since super(dict, self) doesn't work.
    # I'm not sure why, perhaps dict is not a new-style class.

    def __getitem__(self, key):
        for r in self.ranges:
          if key >=r[0] and key<= r[1]:
            return dict.__getitem__(self, self.__keytransform__(r[0]))
        return dict.__getitem__(self, self.__keytransform__(key))

    def __setitem__(self, key, value):
        if not "ranges" in self.__dict__:
            return dict.__setitem__(self, key, value)
        for r in self.ranges:
          if key >=r[0] and key<= r[1]:
            return dict.__setitem__(self, self.__keytransform__(r[0]), value)
        self.ranges.append([ key, key+type(key)(self.err) ])
        return dict.__setitem__(self, self.__keytransform__(key), value)

    def set_err(self, err):
        self.err = err

    def add_range(self, key_min, key_max):
        self.ranges.append([key_min, key_max])

    def __delitem__(self, key):
        return dict.__delitem__(self, self.__keytransform__(key))

    def __contains__(self, key):
        return dict.__contains__(self, self.__keytransform__(key))


class Predictor:
    def __init__(self, points_per_network, W, layers, step, max_steps, acc_value = 1):
      self.points_per_network = points_per_network
      self.layers = layers
      self.step = step
      self.max_steps = max_steps
      self.neural = NeuralLinearComposedNetwork(points_per_network, W, layers, step, max_steps);
      self.analyzer = FactorAnalyzer()
      self.W = W
      self.acc_value = acc_value
      self.acc_x = [ ]
      self.acc_y = [ ]
      self.classificator = Classificator()
      self.alias_dict = FuzzyDict()
      self.do_x_alias = True
      self.do_y_alias = True
      self.back_alias_dict =FuzzyDict()
      self.stamp_width = 0
      self.current_stamp = 0
    def reinit(self):
      self.neural = NeuralLinearComposedNetwork(self.points_per_network, self.W, self.layers, self.step, self.max_steps);
      self.analyzer = FactorAnalyzer()
      self.acc_x = [ ]
      self.acc_y = [ ]
      self.classificator = Classificator()
      self.current_stamp = 0
    def set_alias( self, key, value):
      self.alias_dict[key] = value
      self.back_alias_dict[value] = key
    def get_aliases(self):
      return self.alias_dict
    def set_step_multipliers(self, step_multiplier, step_multiplier_local_opt):
      self.neural.set_multipliers(step_multiplier, step_multiplier_local_opt)
    def get_error(self, X, Y, diff_func_x = None, diff_func_y = None):
        Yout = [ ]
        P = [ ]
        _classes = [ ]
        ll = len(X)
        p = self
        p.predict_p_classes([ X[0] ], Yout, P, ll, _classes)
        errs = 0
        for _p in range(0, len(P)):
            if diff_func_x != None:
                errs += diff_func_x( P[_p], X[(_p) % len(X)])
            else:
                if P[_p] != X[(_p) % len(X)]:
                    errs+=1
            for i in range(0, len(P[_p])):
                if diff_func_y != None:
                    errs+= diff_func_y(Y[_p  % len(Y)][i], Yout[_p][i])
                else:
                    if Y[_p  % len(Y)][i] != Yout[_p][i]:
                        errs+=1
        return errs
    def getBestStamp(self, X, Y, diff_func_x = None, diff_func_y = None, max_stamp = None):
        predictor = self
        stamp_width = 0
        err_stamp = [ ]
        _max = len(X)
        if max_stamp != None:
            _max = max_stamp
        while stamp_width < _max:
            predictor.reinit()
            p = predictor
            p.stamp_width = stamp_width
            p._study(X, Y)
            errs = self.get_error(X, Y, diff_func_x, diff_func_y)
            err_stamp.append([ errs, stamp_width ])
            stamp_width += 1
        predictor.reinit()
        err_stamp.sort()
        print ( err_stamp )
        print ( "Best stamp is ", err_stamp[0][1])
        return err_stamp[0][1]
    def study(self, _X, _Y, check = False, diff_func_x = None, diff_func_y = None):
        if check == True:
            self.stamp_width = self.getBestStamp(_X, _Y)
        self._study(_X, _Y)
    def _study(self, _X, _Y):
      X = copy.deepcopy(_X)
      Y = copy.deepcopy(_Y)
      for i in range(0, len(_X)):
        if self.stamp_width > 0:
            X[i].append(self.current_stamp)
            self.current_stamp = (self.current_stamp + 1) % self.stamp_width
        for j in range(0, len(self.W)):
          if _X[i][j] in self.alias_dict:
            X[i][j] = self.alias_dict[_X[i][j]]
          if _Y[i][j] in self.alias_dict:
            Y[i][j] = self.alias_dict[_Y[i][j]]
      for i in range(0, len(X)):
        self.acc_x.append(X[i])
        self.acc_y.append(Y[i])
      if len(self.acc_x) < self.acc_value:
        return
      acc = [ ]
      for x in range(0, len(self.acc_x)):
        a = copy.deepcopy(self.acc_x[x])
        a.extend(self.acc_y[x])
        acc.append(a)
      self.classificator.reinit(acc)
      self.neural.study(self.acc_x, self.acc_y)
      for x in range(0, len(self.acc_x)):
        self.analyzer.analyze_factor(str(self.acc_x[x]), generate_entity_x, str(self.acc_y[x]))
      self.acc_x = [ ]
      self.acc_y = [ ]
    def all_and(self, arr):
      res = True
      for a in arr:
        res = res and a
      return res
    def some(self, arr):
      res = True
      for a in arr:
        if a == True:
          return True
      return False

    def PrintFactors(self):
      self.analyzer.Print()

    def predict_p(self, prefix, Y, P, depth, is_prefix_time = None):
      _classes = [ ]
      self.predict_p_classes(prefix, Y, P, depth, _classes, is_prefix_time)
      return _classes

    def predict_p_classes(self, _prefix, Y, P, depth, classes, is_prefix_time = None):
      prefix = copy.deepcopy(_prefix)
      for i in range(0, len(_prefix)):
        if self.stamp_width > 0:
            prefix[i].append(self.current_stamp)
            self.current_stamp = (self.current_stamp + 1) % self.stamp_width
        for j in range(0, len(_prefix[i])):
          if _prefix[i][j] in self.alias_dict:
            prefix[i][j] = self.alias_dict[_prefix[i][j]]
      if is_prefix_time == None:
        is_prefix_time = False if self.some(self.neural.cyclic) == True else True
      prefix_vector = [ str(e)  for e in prefix ]
      if is_prefix_time == False or len(_prefix)==0:
        res = self.analyzer.deduct(prefix_vector, depth, generate_entity_x )
        prefix = [ ]
        for r in res:
          prefix.append(eval(r.data))
        for p in prefix:
          Yout = [ 0 for i in range(0, len(self.W)) ]
          appr_p = self.neural.calc_y2( p, Yout)
          Y.append(Yout)
          P.append(p)
          if classes != None:
              acc = copy.deepcopy(appr_p)
              acc.extend(Yout)
              classes.append(self.classificator.classify(acc))
        if res == None or depth > len(res):
          if res == None:
            res = [ ]
          for x in range(int(prefix[len(prefix)-1][0]), int(prefix[len(prefix)-1][0]+depth-len(res))):
            Yout = [ 0 for i in range(0, len(self.W)) ]
            appr_p = self.neural.calc_y2([ x for i in range(0, len(self.W)) ], Yout)
            Y.append(Yout)
            P.append([ x for i in range(0, len(self.W)) ])
            if classes != None:
              acc = copy.deepcopy(appr_p)
              acc.extend(Yout)
              classes.append(self.classificator.classify(acc))
          for i in range(0, len(P)):
            for j in range(0, len(self.W)):
              if int(round(P[i][j],0)) in self.back_alias_dict and self.do_x_alias == True:
                P[i][j] = self.back_alias_dict[int(round(P[i][j],0))]
              else:
                print ("P:Could not find {0} in back_alias_dict {1}", P[i][j], self.back_alias_dict)
              if int(round(Y[i][j],0)) in self.back_alias_dict and self.do_y_alias == True:
                Y[i][j] = self.back_alias_dict[int(round(Y[i][j],0))]
              else:
                print ("Y:Could not find {0} in back_alias_dict {1}", Y[i][j], self.back_alias_dict)
          for i in range(0, len(Y)):
              if self.stamp_width > 0:
                  P[i] = P[i][:len(P[i])-1]
          return
          if classes != None:
              acc = copy.deepcopy(appr_p)
              acc.extend(Yout)
              classes.append(self.classificator.classify(acc))
      else:
        for p in prefix:
          Yout = [ 0 for i in range(0, len(self.W)) ]
          appr_p = self.neural.calc_y2( p, Yout)
          Y.append(Yout)
          P.append(p)
          if classes != None:
              acc = copy.deepcopy(appr_p)
              acc.extend(Yout)
              classes.append(self.classificator.classify(acc))
        X = [ ]
        r = [ ]
        _prefix = prefix[len(prefix)-1]
        XVec = copy.deepcopy(_prefix)
        for x in range(0, depth):
          for j in range(0, len(_prefix)):
            XVec[j]+=1
          X.append(copy.deepcopy(XVec))
        for x in X:
          Yout = [ 0 for i in range(0, len(self.W)) ]
          appr_p = self.neural.calc_y2( x, Yout)
          Y.append(Yout)
          P.append(x)
          if classes != None:
              acc = copy.deepcopy(appr_p)
              acc.extend(Yout)
              classes.append(self.classificator.classify(acc))
      for i in range(0, len(Y)):
        for j in range(0, len(Y[i])):
           if int(round(P[i][j],0)) in self.back_alias_dict and self.do_x_alias == True:
              P[i][j] = self.back_alias_dict[int(round(P[i][j],0))]
           else:
              print ("P:Could not find {0} in back_alias_dict {1}", P[i][j], self.back_alias_dict)
           if int(round(Y[i][j],0)) in self.back_alias_dict and self.do_y_alias == True:
              Y[i][j] = self.back_alias_dict[int(round(Y[i][j],0))]
           else:
              print ("Y:Could not find {0} in back_alias_dict {1}", Y[i][j], self.back_alias_dict)
      for i in range(0, len(Y)):
          if self.stamp_width > 0:
              P[i] = P[i][:len(P[i])-1]


def quadraticTest():
    print ( "Quadratic test begin" )
    W = [ 1.0, 1.0 ]
    x0 = [ 1.0, 2.0 ]
    y0 = [ 9.0, 9.0 ]
    x1 = [ 5.0, 6.0 ]
    y1 = [ 121.0, 121.0 ]
    x2 = [ 9.0, 10.0 ]
    y2 = [ 19.0*19.0, 19.0*19.0 ]
    x3 = [ 13.0, 14.0 ]
    y3 = [ 27.0*27.0, 27.0*27.0 ]
    X = [ x0, x1, x2, x3 ]
    Y = [ y0, y1, y2, y3 ]
    X.append([ 20.0, 20.0 ])
    Y.append([ 1600.0, 1600.0 ])
#    Y.append([ 36.0, 36.0 ])
    P = [ ]
#    X = [ y0, y1, y2 ]
#    Y = [ x0, x1, x2 ]
    grad = [ 0, 0 ]
    Wout = [ W[0], W[1] ]
    step = [ 0.01, 0.01 ]
    p = Predictor(1, Wout, 3, step, 1000000)
    p.study(X, Y)
    Yout = [ ]
    p.predict_p(X, Yout, P, 0)
    print ( "Approximated Y:", Yout )
    print ( "Approximated X:", P )
    print ( "Y:", Y )
    print ( "X:", X )
    print ( "Quadratic test end" )
    for p in range(0, len(P)):
      print ( P[p] )
      print ( Yout[p][0], pow(P[p][0] +P[p][1],2) )
      if abs(Yout[p][0] - pow(P[p][0] +P[p][1],2)) > 1:
        return False
    print ( "Quadratic test PASSED" )
    return True
    #p.neural.pool.wait_ready()
    #p.neural.pool.stop()

def quadraticTest2():
    print ( "Quadratic test2 begin" )
    W = [ 1.0, 1.0 ]
    x0 = [ 1.0, 2.0 ]
    y0 = [ 9.0, 9.0 ]
    x1 = [ 5.0, 6.0 ]
    y1 = [ 121.0, 121.0 ]
    x2 = [ 9.0, 10.0 ]
    y2 = [ 19.0*19.0, 19.0*19.0 ]
    x3 = [ 13.0, 14.0 ]
    y3 = [ 27.0*27.0, 27.0*27.0 ]
    X = [ x0, x1, x2, x3 ]
    Y = [ y0, y1, y2, y3 ]
    X.append([ 20.0, 20.0 ])
    Y.append([ 1600.0, 1600.0 ])
#    Y.append([ 36.0, 36.0 ])
    P = [ ]
#    X = [ y0, y1, y2 ]
#    Y = [ x0, x1, x2 ]
    grad = [ 0, 0 ]
    Wout = [ W[0], W[1] ]
    step = [ 0.01, 0.01 ]
    p = Predictor(1, Wout, 3, step, 1000000)
    p.study(X, Y)
    Yout = [ ]
    p.predict_p([], Yout, P, 6)
    print ( "Approximated Y:", Yout )
    print ( "Approximated X:", P )
    print ( "Y:", Y )
    print ( "X:", X )
    print ( "Quadratic test end" )
    for p in range(0, len(P)-1):
      print ( P[p] )
      print ( Yout[p][0], pow(P[p][0] +P[p][1],2) )
      if abs(Yout[p][0] - pow(P[p][0] +P[p][1],2)) > 1:
        return False
    print ( "Quadratic test2 PASSED" )
    return True
    #p.neural.pool.wait_ready()
    #p.neural.pool.stop()

def periodicTest():
    P = [ ]
#    X = [ y0, y1, y2 ]
#    Y = [ x0, x1, x2 ]
    W = [ 1.0, 1.0 ]
    grad = [ 0, 0 ]
    Wout = [ W[0], W[1] ]
    step = [ 0.01, 0.01 ]
    print ( "Periodic(sin) test begin" )
    p2 = Predictor(1, Wout, 3, step, 1000000)
    Y = [ [math.sin(i), math.sin(i)] for i in range(0,10) ]
    X = [ [ i, i ] for i in range(0,10) ]
    print ( Y, [ [i, i] for i in range(0,10) ] )
    p2.set_step_multipliers(8, 16)
    p2.study([ [i, i] for i in range(0,10) ], Y)
    Yout = [ ]
    P = [ ]
    _classes = p2.predict_p([ [ 0, 0 ] ], Yout, P, 10)
    for c in range(0, len(_classes)):
      if _classes[c] != None:
        print ( "P: ", P[c], Yout[c], "Class: ", _classes[c], _classes[c].vec )
      else:
        print ( "P: ", P[c], Yout[c], "Class: None" )
    print ( "Approximated Sin:", Yout )
    print ( "Approximated X:", P )
    print ( "Sin(X): ", Y )
    print ( "X: ", X )
    print ( "####" )
    print ( "Periodic(sin) test end" )
    for p in range(0, len(P)):
      print ( P[p] )
      print ( Yout[p][0], P[p][0], math.sin(float(P[p][0])) )
      if abs(Yout[p][0] - math.sin(float(P[p][0]))) > 0.1:
        return False
    print ( "Periodic test PASSED" )
    return True
    #p2.neural.pool.wait_ready()
    #p2.neural.pool.stop()

def periodicRandTest():
    P = [ ]
#    X = [ y0, y1, y2 ]
#    Y = [ x0, x1, x2 ]
    W = [ 1.0, 1.0 ]
    grad = [ 0, 0 ]
    Wout = [ W[0], W[1] ]
    step = [ 0.01, 0.01 ]
    print ( "Periodic(sin+rand) test begin" )
    p2 = Predictor(1, Wout, 3, step, 1000000)
    y= [ math.sin(i)+random.randint(0,100) for i in range(0,10) ]
    Y = [ [y[i],y[i]] for i in range(0,10) ]
    X = [ [ i, i ] for i in range(0,10) ]
    print ( Y, [ [i, i] for i in range(0,10) ] )
    p2.set_step_multipliers(8, 16)
    p2.study([ [i, i] for i in range(0,10) ], Y)
    Yout = [ ]
    P = [ ]
    p2.predict_p(X, Yout, P, 0)
    print ( "Approximated Sin+rand:", Yout )
    print ( "Approximated X:", P )
    print ( "Sin(X)+rand: ", Y )
    print ( "X: ", X )
    print ( "####" )
    print ( "Periodic(sin) test end" )
    for p in range(0, len(P)):
      print ( P[p] )
      print ( Yout[p][0], Y[p][0], P[p][0] )
      if abs(Yout[p][0] - Y[p][0]) > 0.5:
        return False
    print ( "Periodic sin+rand test PASSED" )
    return True
    #p2.neural.pool.wait_ready()
    #p2.neural.pool.stop()

def logicTest():
    P = [ ]
    Y = [ ]
    X = [ ]
#    X = [ y0, y1, y2 ]
#    Y = [ x0, x1, x2 ]
    W = [ 1.0, 1.0 ]
    grad = [ 0, 0 ]
    Wout = [ W[0], W[1] ]
    step = [ 0.01, 0.01 ]
    _classes = [ ]
    print ( "Logic test begin" )
    p3 = Predictor(2, Wout, 3, step, 1000000)
    p3.study( [ [ 1.0, 0.0 ] ], [ [ 1.0, 1.0 ] ] )
    X.append( [ 1.0, 0.0 ] )
    Y.append( [ 1.0, 1.0 ] )
    p3.study( [ [ 1.0, 1.0 ] ], [ [ 2.0, 2.0 ] ] )
    X.append( [ 1.0, 1.0 ] )
    Y.append( [ 2.0, 2.0 ] )
    p3.study( [ [ 1.0, 1.0 ] ], [ [ 2.0, 2.0 ] ] )
    X.append( [ 1.0, 1.0 ] )
    Y.append( [ 2.0, 2.0 ] )
    p3.study( [ [ 0.0, 0.0 ] ], [ [ 0.0, 0.0 ] ] )
    X.append( [ 0.0, 0.0 ] )
    Y.append( [ 0.0, 0.0 ] )
    p3.study( [ [ 0.0, 1.0 ] ], [ [ 1.0, 1.0 ] ] )
    X.append( [ 0.0, 1.0 ] )
    Y.append( [ 1.0, 1.0 ] )
    p3.study( [ [ 1.0, 0.0 ] ], [ [ 1.0, 1.0 ] ] )
    X.append( [ 1.0, 0.0 ] )
    Y.append( [ 1.0, 1.0 ] )
    p3.study( [ [ 0.0, 1.0 ] ], [ [ 1.0, 1.0 ] ] )
    X.append( [ 0.0, 1.0 ] )
    Y.append( [ 1.0, 1.0 ] )
    p3.study( [ [ 1.0, 0.0 ] ], [ [ 1.0, 1.0 ] ] )
    X.append( [ 1.0, 0.0 ] )
    Y.append( [ 1.0, 1.0 ] )
    p3.study( [ [ 1.0, 1.0 ] ], [ [ 2.0, 2.0 ] ] )
    X.append( [ 1.0, 1.0 ] )
    Y.append( [ 2.0, 2.0 ] )
    p3.study( [ [ 1.0, 1.0 ] ], [ [ 2.0, 2.0 ] ] )
    X.append( [ 1.0, 1.0 ] )
    Y.append( [ 2.0, 2.0 ] )
    p3.study( [ [ 0.0, 0.0 ] ], [ [ 0.0, 0.0 ] ] )
    X.append( [ 0.0, 0.0 ] )
    Y.append( [ 0.0, 0.0 ] )
    p3.study( [ [ 0.0, 1.0 ] ], [ [ 1.0, 1.0 ] ] )
    X.append( [ 0.0, 1.0 ] )
    Y.append( [ 1.0, 1.0 ] )
    p3.study( [ [ 1.0, 0.0 ] ], [ [ 1.0, 1.0 ] ] )
    X.append( [ 1.0, 0.0 ] )
    Y.append( [ 1.0, 1.0 ] )
    Yout = [ ]
    P = [ ]
    p3.predict_p_classes(X, Yout, P, 0, _classes)
    print ( "Approximated Y: ", Yout )
    print ( "Approximated X: ", P )
    print ( "Y:", Y )
    print ( "X:", X )
    print ( "Logic test end" )
    for c in range(0, len(_classes)):
      if _classes[c] != None:
        print ( "P: ", P[c], Yout[c], "Class: ", _classes[c], _classes[c].vec )
      else:
        print ( "P: ", P[c], Yout[c], "Class: None" )
    for p in range(0, len(X)):
      print ( P[p], len(P), len(X) )
      print ( Yout[p], Y[p], P[p] )
      if abs(Yout[p][0] - Y[p][0]) > 1.0:
        return False
    print ( "Logic test PASSED" )
    return True
    #p3.neural.pool.wait_ready()
    #p3.neural.pool.stop()

def logicTest2():
    P = [ ]
    Y = [ ]
    X = [ ]
#    X = [ y0, y1, y2 ]
#    Y = [ x0, x1, x2 ]
    W = [ 1.0, 1.0 ]
    grad = [ 0, 0 ]
    Wout = [ W[0], W[1],W[0], W[1], W[0], W[1],W[0], W[1] ]
    step = [ 0.01, 0.01,0.01, 0.01,0.01, 0.01,0.01, 0.01 ]
    _classes = [ ]
    print ( "Logic test begin" )
    p3 = Predictor(1, Wout, 3, step, 1000000)
    print ( "Predictor created" )
    X.append([ 0, 1, 1, 0, 0, 1, 1, 0])
    Y.append([ 1, 0, 0, 0, 0, 0, 0, 0])
    X.append([ 1, 0, 0, 1, 1, 0, 0, 1])
    Y.append([ 0, 0, 0, 0, 0, 0, 0, 1])
    X.append([ 1, 0, 1, 1, 1, 1, 0, 1])
    Y.append([ 0, 1, 1, 1, 1, 1, 1, 0])
    print ( "Before study" )
    p3.study(X, Y)
    print ( "After study" )
    Yout = [ ]
    P = [ ]
    p3.classificator.print_info()
    p3.predict_p_classes(X, Yout, P, 0, _classes, False)
    print ( "Approximated Y: ", Yout )
    print ( "Approximated X: ", P )
    print ( "Y:", Y )
    print ( "X:", X )
    print ( "Logic test end" )
    for c in range(0, len(_classes)):
      if _classes[c] != None:
        print ( "P: ", P[c], Yout[c], "Class: ", _classes[c], _classes[c].vec )
      else:
        print ( "P: ", P[c], Yout[c], "Class: None" )
    for c in range(0, len(_classes)):
      if _classes[c] != None:
        print ( "P: ", P[c], Yout[c], "Class: ", _classes[c], _classes[c].vec )
      else:
        print ( "P: ", P[c], Yout[c], "Class: None" )
    for p in range(0, len(X)):
      print ( P[p], len(P), len(X) )
      print ( Yout[p], Y[p], P[p] )
      if abs(Yout[p][0] - Y[p][0]) > 1.0:
        return False
    print ( "Logic test2 PASSED" )
    return True
#    p3.neural.pool.wait_ready()
#    p3.neural.pool.stop()

def classifierTest():
    print ( "Classifier test begin" )
    circle = [ ]
    a = 0
    while a < 360:
      x = 10*math.cos(a)
      y = 10*math.sin(a)
      circle.append([x,y])
      a+=30

    square = [ ]
    square.append([0,0])
    square.append([0,5])
    square.append([5,5])
    square.append([5,0])
    c1 = Cluster(square, None, "square")
    print ( circle )
    c2 = Cluster(circle, None, "circle")
    print ( circle )
    c = Classificator()
    c.add_cluster(c1)
    c.add_cluster(c2)
    print ( circle )
    res = c.classify_vec(square)
    if res:
      print ( "Found:", res, res.name )
    else:
      print ( "Not found" )
    print ( square )
    if res.name != "square":
      return False
    res = c.classify_vec(circle)
    if res:
      print ( "Found:", res, res.name )
    else:
      print ( "Not found" )
    print ( circle )
    if res.name != "circle":
      return False
    circle.extend(square)
    res = c.classify_vec(circle, False, False)
    found_circle = False
    found_square = False
    for r in res:
      print ( "Found:", r[0], r[0].name )
      if r[0].name == "circle":
        found_circle = True
      if r[0].name == "square":
        found_square = True
    if (found_circle and found_square) != True:
      return False
    c.print_info()
    print ( "Classifier test end PASSED" )
    return True

def classifierTest2():
    print ( "Classifier2 test begin" )
    c = Classificator()
    X = [ ]
    Y = [ ]
    X.append([ 0, 1, 1, 0, 0, 1, 1, 0])
    Y.append([ 1, 0, 0, 0, 0, 0, 0, 0])
    X[0].extend(Y[0])
    X.append([ 1, 0, 0, 1, 1, 0, 0, 1])
    Y.append([ 0, 0, 0, 0, 0, 0, 0, 1])
    X[1].extend(Y[1])
    c.reinit(X)
    c.print_info()
    if len(c.clusters) != 2:
      return False
    print ( "Classifier2 test end PASSED" )
    return True

def weatherTest():
    Wout = [ 1.0 ]
    step = [ 0.1 ]
    p = Predictor(1, Wout, 3, step, 1000000)
    SUN_SHINE = 0
    RAIN = 1
    NAN = 0
    X = [ ["SUN_SHINE"], ["SUN_SHINE"], ["RAIN"], ["RAIN"], ["RAIN"], ["SUN_SHINE"], ["RAIN"] ]
    Y = copy.deepcopy(X)
    p.set_alias("SUN_SHINE", 0)
    p.set_alias("RAIN", 1)
    p.study(X,Y)
    Yout = [ ]
    P = [ ]
    _classes = [ ]
    p.predict_p_classes([ ], Yout, P, 700, _classes)
    print ( "Approximated Y: ", Yout )
    print ( "Approximated X: ", P )
    print ( "Y:", Y )
    print ( "X:", X )
    for c in range(0, len(_classes)):
      if _classes[c] != None:
        print ( "P: ", P[c], Yout[c], "Class: ", _classes[c], _classes[c].vec )
      else:
        print ( "P: ", P[c], Yout[c], "Class: None" )
    if len(P) < 10:
      return False
    sun_shine = 0
    rain = 0
    for p in range(0, len(P)):
      if P[p][0] == "SUN_SHINE":
        sun_shine+=1
      if P[p][0] == "RAIN":
        rain+=1
      if P[p][0] != Yout[p][0]:
        return False
    if sun_shine <200:
      print ( "error:sun_shine ", sun_shine )
      print ( "rain ", rain )
      return False
    if rain<350:
      print ( "error: rain ", rain )
      print ( "sun_shine ", sun_shine )
      return False
    print ( "weatherTest PASSED" )
    return True

def weatherTest2():
    Wout = [ 1.0, 1.0, 1.0, 1.0 ]
    step = [ 0.1, 0.1, 0.1, 0.1 ]
    p = Predictor(1, Wout, 3, step, 1000000)
    p.set_alias("SUN_SHINE", 0)
    p.set_alias("RAIN", 1)
    p.set_alias("WARM", 2)
    p.set_alias("COLD", 3)
    p.set_alias("WINTER", 40)
    p.set_alias("SUMMER", 50)
    p.set_alias("GOOD_WEATHER", 6)
    p.set_alias("BAD_WEATHER", 7)
    X = [ ]
    Y = [ ]
    X.extend([ ["SUN_SHINE", "COLD", "WINTER", "GOOD_WEATHER"] ])
    Y.extend([ [ -20.0 for i in range(0, 4) ] ])
    X.extend([ ["SUN_SHINE", "WARM", "WINTER", "BAD_WEATHER"] ])
    Y.extend([ [ 5.0 for i in range(0, 4) ] ])
    X.extend([ ["SUN_SHINE", "COLD", "WINTER", "GOOD_WEATHER"] ])
    Y.extend([ [ -20.0 for i in range(0, 4) ] ])
    X.extend([ ["SUN_SHINE", "WARM", "WINTER", "BAD_WEATHER"] ])
    Y.extend([ [ 5.0 for i in range(0, 4) ] ])
    X.extend([ ["SUN_SHINE", "COLD", "SUMMER", "BAD_WEATHER"] ])
    Y.extend([ [ 10.0 for i in range(0, 4) ] ])
    X.extend([ ["RAIN", "WARM", "SUMMER", "GOOD_WEATHER"] ])
    Y.extend([ [ 20.0 for i in range(0, 4) ] ])
    X.extend([ ["RAIN", "COLD", "SUMMER", "BAD_WEATHER"] ])
    Y.extend([ [ 8.0 for i in range(0, 4) ] ])
    X.extend([ ["SUN_SHINE", "WARM", "SUMMER", "GOOD_WEATHER"] ])
    Y.extend([ [ 25.0 for i in range(0, 4) ] ])
    X.extend([ ["SUN_SHINE", "WARM", "WINTER", "BAD_WEATHER"] ])
    Y.extend([ [ 5.0 for i in range(0, 4) ] ])
    X.extend([ ["SUN_SHINE", "COLD", "WINTER", "GOOD_WEATHER"] ])
    Y.extend([ [ -20.0 for i in range(0, 4) ] ])
    p.study(X,Y)
    Yout = [ ]
    P = [ ]
    _classes = [ ]
    p.predict_p_classes([ [2000, "COLD", 20000, "BAD_WEATHER"] ], Yout, P, 4, _classes)
    print ( "Approximated Y: ", Yout )
    print ( "Approximated X: ", P )
    print ( "Y:", Y )
    print ( "X:", X )
    for c in range(0, len(_classes)):
      if _classes[c] != None:
        print ( "P: ", P[c], Yout[c], "Class: ", _classes[c], _classes[c].vec )
      else:
        print ( "P: ", P[c], Yout[c], "Class: None" )
    if len(P) != 5:
      print ( "Wrong P, should be 5", len(P) )
      return False
    if P[0][2] != "SUMMER" or abs(Yout[0][0] - 10.0) > 5:      #the only valid solution is SUMMER
      print ( "Wrong guess ", P[0][2] )
      return False
    if len(p.classificator.clusters) != 6:
      print ( "Wrong classes number", len(p.classificator.clusters) )
      for c in p.classificator.clusters:
        print ( str(c) )
      return False
    Yout = [ ]
    P = [ ]
    _classes = [ ]
    p.predict_p_classes([ ["SUN_SHINE", "WARM", 500000, "BAD_WEATHER"] ], Yout, P, 2, _classes)
    print ( "Approximated Y: ", Yout )
    print ( "Approximated X: ", P )
    print ( "Y:", Y )
    print ( "X:", X )
    for c in range(0, len(_classes)):
      if _classes[c] != None:
        print ( "P: ", P[c], Yout[c], "Class: ", _classes[c], _classes[c].vec )
      else:
        print ( "P: ", P[c], Yout[c], "Class: None" )
    if len(P) != 3:
      print ( "Wrong P, should be 3", len(P) )
      return False
    if P[0][2] != "WINTER" or abs(Yout[0][0] - 5.0) > 1:        #the only valid solution is WINTER
      print ( "Wrong guess ", P[0][2] )
      return False
    if len(p.classificator.clusters) != 6:
      print ( "Wrong classes number", len(p.classificator.clusters) )
      for c in p.classificator.clusters:
        print ( str(c) )
      return False
    print ( "weatherTest2 PASSED" )
    return True

def psychoTest():
    Wout = [ 1.0 ]
    step = [ 0.1 ]
    p = Predictor(1, Wout, 3, step, 1000000)
    p.set_alias("HEAD_ACHE", 0)
    p.set_alias("FLU", 200)
    p.set_alias("HEART_RACE", 2)
    p.set_alias("SWEAT", 3)
    p.set_alias("ANXIETY", 40)
    p.set_alias("CALM", 500)
    p.set_alias("COLD", 300)
    p.set_alias("SPORT", 4)
    p.set_alias("GOOD", 600)
    p.set_alias("FEAR", 44)
    p.set_alias("BAD", 700)
    X = [ ]
    Y = [ ]
    X.extend([ ["FLU"], ["HEART_RACE"], ["FEAR"], ["BAD"] ])
    Y.extend([ [ -20.0] for i in range(0, 4)  ])
    X.extend([ ["BAD"], ["FLU"], ["FEAR"], ["HEART_RACE"] ])
    Y.extend([ [ -20.0] for i in range(0, 4)  ])
    X.extend([ ["HEART_RACE"], ["SWEAT"], ["FEAR"], ["ANXIETY"] ])
    Y.extend([ [ -20.0] for i in range(0, 4) ])
    X.extend([ ["CALM"], ["GOOD"], ["SWEAT"], ["SPORT"] ])
    Y.extend([ [ -20.0 ] for i in range(0, 4) ])
    p.study(X,Y)
    Yout = [ ]
    P = [ ]
    _classes = [ ]
    p.predict_p_classes([ ["COLD"], ["BAD"] ], Yout, P, 4, _classes)
    p.PrintFactors()
    print ( "Approximated Y: ", Yout )
    print ( "Approximated X: ", P )
    print ( "Y:", Y )
    print ( "X:", X )
    for c in range(0, len(_classes)):
      if _classes[c] != None:
        print ( "P: ", P[c], Yout[c], "Class: ", _classes[c], _classes[c].vec )
      else:
        print ( "P: ", P[c], Yout[c], "Class: None" )
    print ( "psychoTest PASSED" )
    return True

def getBestStamp(predictor, X, Y, max_stamp = None):
    stamp_width = 0
    err_stamp = [ ]
    _max = len(X)
    if max_stamp != None:
        _max = max_stamp
    while stamp_width < _max:
        predictor.reinit()
        p = predictor
        p.stamp_width = stamp_width
        p.study(X, Y)
        Yout = [ ]
        P = [ ]
        _classes = [ ]
        ll = len(X)
        p.predict_p_classes([ X[0] ], Yout, P, ll, _classes)
        errs = 0
        for _p in range(0, len(P)):
            if P[_p] != X[(_p) % len(X)]:
                errs+=1
            for i in range(0, len(P[_p])):
                    if (float(Y[_p  % len(Y)][i]) != 0) and (float(Yout[_p][i]) != 0):
                        if float(Y[_p  % len(Y)][i])/float(Y[_p  % len(Y)][i]) != float(Yout[_p][i])/float(Yout[_p][i]):
                            errs+=1
        err_stamp.append([ errs, stamp_width ])
        stamp_width += 1
    predictor.reinit()
    err_stamp.sort()
    print ( err_stamp )
    print ( "Best stamp is ", err_stamp[0][1])
    return err_stamp[0][1]


def stockTest():
    Wout = [ 1.0, 1.0, 1.0, 1.0 ]
    step = [ 0.1, 0.1, 0.1, 0.1 ]
    p = Predictor(1, Wout, 3, step, 1000000)
    p.set_alias("NASDAQ_DOWN", 0)
    p.set_alias("NASDAQ_UP", 1)
    p.set_alias("DOW_DOWN", 2)
    p.set_alias("DOW_UP", 3)
    p.set_alias("S&P_DOWN", 4)
    p.set_alias("S&P_UP", 5)
    p.set_alias("NYSE_DOWN", 6)
    p.set_alias("NYSE_UP", 7)
    p.do_y_alias = False
    X = [ ]
    Y = [ ]
    _X = [ ["NASDAQ_DOWN", "DOW_UP", "S&P_UP", "NYSE_DOWN"] ]
    X.extend(_X)
    Y.extend([ [ -5, 10, 10, -5 ] ])
    _X = [ [ "NASDAQ_UP", "DOW_DOWN", "S&P_DOWN", "NYSE_DOWN"] ]
    X.extend(_X)
    Y.extend([ [ 20, -10, -10, -5 ] ])
    _X = [ [ "NASDAQ_DOWN", "DOW_DOWN", "S&P_DOWN", "NYSE_DOWN"] ]
    X.extend(_X)
    Y.extend([ [ -5, -10, -10, -5 ] ])
    _X = [ [ "NASDAQ_UP", "DOW_DOWN", "S&P_DOWN", "NYSE_UP"] ]
    X.extend(_X)
    Y.extend([ [ 15, -8, -8, 10 ] ])
    _X = [ [ "NASDAQ_DOWN", "DOW_UP", "S&P_UP", "NYSE_DOWN"] ]
    X.extend(_X)
    Y.extend([ [ -5, 10, 10, -12 ] ])
    _X = [ ["NASDAQ_DOWN", "DOW_DOWN", "S&P_UP", "NYSE_DOWN"] ]
    X.extend(_X)
    Y.extend([ [ -5, -10, -10, -5 ] ])
    _X = [ ["NASDAQ_UP", "DOW_DOWN", "S&P_DOWN", "NYSE_DOWN"] ]
    X.extend(_X)
    Y.extend([ [ 15, -8, -8, 10 ] ])
    _X = [ ["NASDAQ_UP", "DOW_UP", "S&P_DOWN", "NYSE_UP"] ]
    X.extend(_X)
    Y.extend([ [ 20, -10, -10, -5 ] ])
    _X = [ ["NASDAQ_UP", "DOW_DOWN", "S&P_DOWN", "NYSE_DOWN"] ]
    X.extend(_X)
    Y.extend([ [ 20, -10, -10, -5 ] ])
    _X = [ ["NASDAQ_DOWN", "DOW_DOWN", "S&P_DOWN", "NYSE_DOWN"] ]
    X.extend(_X)
    Y.extend([ [ -5, -10, -10, -5 ] ])
    _X = [ ["NASDAQ_UP", "DOW_DOWN", "S&P_DOWN", "NYSE_UP"] ]
    X.extend(_X)
    Y.extend([ [ 15, -8, -8, 10 ] ])
    _X = [ ["NASDAQ_DOWN", "DOW_UP", "S&P_UP", "NYSE_DOWN"] ]
    X.extend(_X)
    Y.extend([ [ -5, 10, 10, -12 ] ])
    p.study(X,Y, True)
    Yout = [ ]
    P = [ ]
    _classes = [ ]
    ll = len(X)
    p.predict_p_classes([ X[0] ], Yout, P, ll, _classes)
    print ( "Approximated Y: ", Yout )
    print ( "Approximated X: ", P )
    print ( "Y:", Y )
    print ( "X:", X )
    errs = 0
    errs2 = 0
    for _p in range(0, len(P)):
        if P[_p] != X[(_p) % len(X)]:
            print ("Wrong prediction", P[_p], X[(_p) % len(X)])
            errs+=1
        for i in range(0, len(P[_p])):
            if (float(Yout[_p][i]) != 0):
                if (float(Yout[_p][i]) > 0):
                    print ( P[_p][i], ": BUY")
                if (float(Yout[_p][i]) < 0):
                    print ( P[_p][i], ": SELL")
            if (float(Y[_p  % len(Y)][i]) != 0) and (float(Yout[_p][i]) != 0):
                if float(Y[_p  % len(Y)][i])/float(Y[_p  % len(Y)][i]) != float(Yout[_p][i])/float(Yout[_p][i]):
                    errs2+=1
        print ( Y[_p % len(Y)], Yout[_p] )
        if P[_p] != X[(_p) % len(X)]:
            print ("***********************************")

    print ("Wrong predictions: ", errs, errs2)
    if errs > ll/3:
        return False
    if errs2 > ((ll*4)/3):
        return False
    for c in range(0, len(_classes)):
      if _classes[c] != None:
        print ( "P: ", P[c], Yout[c], "Class: ", _classes[c], _classes[c].vec )
      else:
        print ( "P: ", P[c], Yout[c], "Class: None" )
    if len(P) < 5:
      print ( "Wrong P, should be at least 5", len(P) )
      return False

    if len(p.classificator.clusters) != 9:
      print ( "Wrong classes number", len(p.classificator.clusters) )
      for c in p.classificator.clusters:
        print ( str(c) )
      return False
    print ( "stockTest PASSED" )
    return True



def stockTest2():
    Wout = [ 1.0, 1.0]
    step = [ 0.1, 0.1]
    p = Predictor(1, Wout, 3, step, 1000000)
    p.set_alias("NASDAQ_DOWN", 0)
    p.set_alias("NASDAQ_UP", 1)
    p.set_alias("DOW_DOWN", 2)
    p.set_alias("DOW_UP", 3)
    p.set_alias("S&P_DOWN", 4)
    p.set_alias("S&P_UP", 5)
    p.set_alias("NYSE_DOWN", 6)
    p.set_alias("NYSE_UP", 7)
    X = [ ]
    Y = [ ]
    X.extend([ ["NASDAQ_DOWN", "DOW_UP"] ])
    Y.extend([ [ "S&P_DOWN", "NYSE_DOWN" ] ])
    X.extend([ ["NASDAQ_UP", "DOW_DOWN"] ])
    Y.extend([ [ "S&P_UP", "NYSE_UP" ] ])
    X.extend([ ["NASDAQ_DOWN", "DOW_DOWN"] ])
    Y.extend([ [ "S&P_DOWN", "NYSE_DOWN" ] ])
    X.extend([ ["NASDAQ_UP", "DOW_UP"] ])
    Y.extend([ [ "S&P_DOWN", "NYSE_DOWN" ] ])
    p.study(X,Y)
    Yout = [ ]
    P = [ ]
    _classes = [ ]
    p.predict_p_classes([ ["NASDAQ_DOWN", "DOW_UP" ] ], Yout, P, 3, _classes)
    print ( "Approximated Y: ", Yout )
    print ( "Approximated X: ", P )
    print ( "Y:", Y )
    print ( "X:", X )
    for c in range(0, len(_classes)):
      if _classes[c] != None:
        print ( "P: ", P[c], Yout[c], "Class: ", _classes[c], _classes[c].vec )
      else:
        print ( "P: ", P[c], Yout[c], "Class: None" )
    if len(P) < 3:
      print ( "Wrong P, should be at least 5", len(P) )
      return False
    if P[1][0] != "NASDAQ_UP" or P[1][1] != "DOW_DOWN" or P[2][0] != "NASDAQ_DOWN" or P[2][1] != "DOW_DOWN":
      print ( "Wrong prediction P: ", P )
      return False
    if P[3][0] != "NASDAQ_UP" or P[3][1] != "DOW_UP":
      print ( "Wrong prediction P: ", P )
      return False
    if Yout[1][0] != "S&P_UP" or Yout[1][1] != "NYSE_UP" or Yout[2][0] != "S&P_DOWN" or Yout[2][1] != "NYSE_DOWN":
      print ( "Wrong prediction Y: ", Y )
      return False
    if Yout[3][0] != "S&P_DOWN" or Yout[3][1] != "NYSE_DOWN":
      print ( "Wrong prediction Y: ", Y )
      return False
    Yout = [ ]
    P = [ ]
    _classes = [ ]
    p.predict_p_classes([ ["NASDAQ_UP", "DOW_DOWN" ] ], Yout, P, 3, _classes)
    print ( "Approximated Y: ", Yout )
    print ( "Approximated X: ", P )
    print ( "Y:", Y )
    print ( "X:", X )
    for c in range(0, len(_classes)):
      if _classes[c] != None:
        print ( "P: ", P[c], Yout[c], "Class: ", _classes[c], _classes[c].vec )
      else:
        print ( "P: ", P[c], Yout[c], "Class: None" )
    if P[1][0] != "NASDAQ_DOWN" or P[1][1] != "DOW_DOWN":
      print ( "Wrong prediction P: ", P )
      return False
    if P[2][0] != "NASDAQ_UP" or P[2][1] != "DOW_UP":
      print ( "Wrong prediction P: ", P )
      return False
    if Yout[1][0] != "S&P_DOWN" or Yout[1][1] != "NYSE_DOWN":
      print ( "Wrong prediction Y: ", Y )
      return False
    if Yout[2][0] != "S&P_DOWN" or Yout[2][1] != "NYSE_DOWN":
      print ( "Wrong prediction Y: ", Y )
      return False
    print ( "stockTest2 PASSED" )
    return True


def predict_thread(p, f, f2):
    i = 0
    while(True):
      X = [ ]
      Y = [ ]
      P = [ ]
      line = f.readline()
      depth = int(f.readline())
      s = "[ " + line + " ] "
      v = eval(s)
      X.append(v)
      print ( "Predict:" )
      print ( "X: " )
      print ( X )
      print ( "Y: " )
      print ( Y )
      print ( "P: " )
      print ( P )
      print ( "depth:" )
      print ( depth )
      p.predict_p(X, Y, P, depth)
      f2.write(str(P)+"\n")
      f2.write(str(Y)+"\n")
      f2.flush()

def simpleTest():
    Wout = [ 1.0, 1.0 ]
    step = [ 0.1, 0.1 ]
    p = Predictor(1, Wout, 3, step, 1000000)
    X = [ ]
    Y = [ ]
    X.append([0,1 ])
    Y.append([0,1 ])
    X.append([1,1 ])
    Y.append([0,1 ])
    X.append([1,2 ])
    Y.append([1,0 ])
    X.append([2,2 ])
    Y.append([0,1 ])
    X.append([4,2 ])
    Y.append([1,0 ])
    X.append([3,3 ])
    Y.append([0,1 ])
    X.append([5,3 ])
    Y.append([0,1 ])
    X.append([8,4 ])
    Y.append([1,0 ])
    X.append([20,40 ])
    Y.append([1,0 ])
    X.append([50,100 ])
    Y.append([1,0 ])
    X.append([500,1000 ])
    Y.append([1,0 ])
    X.append([600,1200 ])
    Y.append([1,0 ])
    X.append([700,2000 ])
    Y.append([0,1 ])
    X.append([5000,2300 ])
    Y.append([0,1 ])
    X.append([5000,2500 ])
    Y.append([1,0 ])
    X.append([6000,12000 ])
    Y.append([1,0 ])
    X.append([4000,9000 ])
    Y.append([0,1 ])
    X.append([4000,8000 ])
    Y.append([0,1 ])
    X.append([3000,6000 ])
    Y.append([1,0 ])
    X.append([100,7 ])
    Y.append([0,1 ])
    p.study(X,Y)
    Yout = [ ]
    P = [ ]
    _classes = [ ]
    p.predict_p_classes([[6000,3000],[10,20],[100,200],[5000,10000],[700,1400],[3500,7000],[100, 3]], Yout, P, 0, _classes, True)
    print ( "Approximated Y: ", Yout )
    print ( "Approximated X: ", P )
    print ( "Y:", Y )
    print ( "X:", X )
    for c in range(0, len(_classes)):
      if _classes[c] != None:
        print ( "P: ", P[c], Yout[c], "Class: ", _classes[c], _classes[c].vec )
      else:
        print ( "P: ", P[c], Yout[c], "Class: None" )

    for c in p.classificator.clusters:
        print ( str(c) )

    guessed = 0

    for y in range(0, len(Yout)-1):
      if abs(Yout[y][1])<0.1 and abs(Yout[y][0])>0.1:
        guessed+=1

    if guessed < (len(Yout)-1)/2:
      return False

    print ( "simpleTest PASSED" )
    return True


def run_all_tests(_rep = True):
    mystdout = None
    oldstdout = None
    s = ""
    rep = True
    if rep:
      oldstdout = sys.stdout
    if rep:
      mystdout = StringIO()
      sys.stdout = mystdout
    res = True
#    res = weatherTest()
    all_pass = True
    failed = [ ]
#    if res != True:
#      print ( "weatherTest FAILED!" )
#      failed.append("weatherTest FAILED!")
#      all_pass = False
    res = quadraticTest()
    if res != True:
      print ( "quadraticTest FAILED!" )
      failed.append("quadraticTest FAILED!")
      all_pass = False
    res = quadraticTest2()
    if res != True:
      print ( "quadraticTest2 FAILED!" )
      failed.append("quadraticTest2 FAILED!")
      all_pass = False
    res = periodicTest()
    if res != True:
      print ( "periodicTest FAILED!" )
      failed.append("periodicTest FAILED!")
      all_pass = False
    res = periodicRandTest()
    if res != True:
      print ( "periodicRandTest FAILED!" )
      failed.append("periodicRandTest FAILED!")
      all_pass = False
    res = logicTest()
    if res != True:
      print ( "logicTest FAILED!" )
      failed.append("logicTest FAILED!")
      all_pass = False
    res = classifierTest()
    if res != True:
      print ( "classifierTest FAILED!" )
      failed.append("classifierTest FAILED!")
      all_pass = False
    res = logicTest2()
    if res != True:
      print ( "logicTest2 FAILED!" )
      failed.append("logicTest2 FAILED!")
      all_pass = False
    res = classifierTest2()
    if res != True:
      print ( "classifierTest2 FAILED!" )
      failed.append("classifierTest2 FAILED!")
      all_pass = False
    res = weatherTest2()
    if res != True:
      print ( "weatherTest2 FAILED!" )
      failed.append("weatherTest2 FAILED!")
      all_pass = False
    res = stockTest()
    if res != True:
      print ( "stockTest FAILED!" )
      failed.append("stockTest FAILED!")
      all_pass = False
    res = stockTest2()
    if res != True:
      print ( "stockTest2 FAILED!" )
      failed.append("stockTest2 FAILED!")
      all_pass = False
#    res = simpleTest()
#    if res != True:
#      print ( "simpleTest FAILED!" )
#      failed.append("simpleTest FAILED!")
#      all_pass = False
    if rep:
      s = mystdout.getvalue()
    if all_pass == True:
      s = "All test cases PASSED \n" + s
    else:
      s = "Test cases failed: " + str(failed) + "\n" + s
    if rep:
      sys.stdout = oldstdout
    if _rep == False:
      print ( s )
    return s



def guess_thread():
    i = 0
    X = [ ]
    Y = [ ]
    Wout = [ 1.0, 1.0 ]
    step = [ 0.1, 0.1 ]
    p = Predictor(1, Wout, 3, step, 1000000)
    P = [ ]
    n = 0
    prev_vec = None
    v = None
    while(True):
      line = sys.stdin.readline()
      line = line.replace('\n','')
      line = line.replace('\r','')
      if prev_vec == None:
        prev_vec =[ 0, 0 ]
      s = "[ " + str(prev_vec[1]) +"," + line + " ] "
      print ( "readline ", s )
      if len(P) > 0:
        print ( "my_guess:", P[1][1] )
        if str(P[1][1]) == line:
          print ( "CORRECT: ",n, "out of", i-9 )
          n+=1
      prev_vec = v
      v = eval(s)
      X.append(v)
      Y.append(v)
      p.study(X, Y)
      Yout = [ ]
      P = [ ]
      if i>=9:
        p.predict_p([ v ], Yout, P, 2)
        print ( P )
      i+=1

if __name__ == "__main__":
    if len(sys.argv) > 1:
      if sys.argv[1] == "guess_thread":
        guess_thread()
#    simpleTest()
#    stockTest()
    run_all_tests(False)
    exit(0)
