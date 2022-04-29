# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 12:32:41 2022

@author: Giovanni Artiglio
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import csv

N=10 #vector size
NUMDYADS = 10
NPOPSIZE = 10

class Population :
    ##cosntructs a population
    
    def __init__ (self,b,n,p, a=100, d=10):
        self._budget = int(b)
        self._fList = []
        self._f_counts_list = []
        self._resolution = int(n)
        self._popSize = int(p)
        self._gself = np.zeros(int(n))
        self._atoms = a
        self._nperturbations = d
        
        for i in range(self._popSize):
            f,fc = self.create_f()
            self._fList.append(f)
            self._f_counts_list.append(fc)
            
        self._gself = sum(self._fList)/self._popSize
    
    
    def get_fList (self):
        return self._fList
    
    def get_g(self):
        #returns g
        return self._gself
    
    def show_info (self):
        print("Budget: " + str(self._budget))
        print("n: "+str(self._resolution))
        print("PopSize: " + self._popSize)
        print("fList: " + str(self._fList))
    
    
    def show_P (self):
        fList = self.get_fList()
        

        plt.figure()
                
        for f in fList:
        
            self.plot_f(f)
        
        plt.show()
        
    
    def plot_f(self,f):
        
        f_n = len(f)
        dbl_y = [f[int(i)] for i in np.arange(0,2*f_n)/2.0 ]
        dbl_x = [int(i) for i in np.arange(1,2*f_n+1)/2.0 ]
        plt.plot(dbl_x,dbl_y)
        
        
    def plot_f_sum(self):
        fList = self.get_fList()

        f_tot = sum(fList)/len(fList)
        f_n = len(f_tot)
        dbl_y = [f_tot[int(i)] for i in np.arange(0,2*f_n)/2.0 ]
        dbl_x = [int(i) for i in np.arange(1,2*f_n+1)/2.0 ]
        plt.plot(dbl_x,dbl_y)
        
        
    '''def create_dyad (self):
        b = self._budget
        n=self._resolution
        x1 = np.random.randint(0, b+1)
        x2 = np.random.randint(b+1, n)
        lam = (x2+.5-b)/(x2-x1)
        
        d = np.zeros(self._resolution)
        d[x1] = lam
        d[x2] = 1-lam
        
        return d'''
    
    def create_f (self):
        #return a new f_i vector of size n
        #f distribution atomized
        
        n=self._resolution
        b=self._budget
        d=self._nperturbations
        a=self._atoms
        
        f = np.zeros(n)
        f_counts = np.zeros(n)
        
        #f_counts[0:2*b+1] = int(a/(2*b+1))
        #f_counts[b] += a%(2*b)
        
        target = b*2
        
        f[0:target+1] = a/(target+1)
        f = np.cumsum(f)+np.random.rand()
        
        cumsum = 0
        
        for i in range(0,b+1):
            
            f_counts[i] = int(f[i] - cumsum)
        
            cumsum += f_counts[i]
        
        for i in range(0,b):
            f_counts[target-i] = f_counts[i]
        
        if sum(f_counts) == 99:
            f_counts[b] += 1
            
        if sum(f_counts) == 101:
            f_counts[b] += -1
        
        #print(f_counts)
        for i in range(d):
            f_counts = self.mutate_f_counts(f_counts)
        
        f = f_counts/(a*1.0)
        
        return f,f_counts
    
    def mutate_f_counts (self,f_counts):
        #mutation method: atom permutation
        n=self._resolution
        a=self._atoms
        #f_support = f_counts > 0
        
        #atom sampling
        i = np.random.randint(0,a)
        j = np.random.randint(0,a-1)
        j = (i+j+1)%a
        
        #print(i, np.cumsum(f_counts))
        #print(np.nonzero(np.cumsum(f_counts) > i))
        x1 = np.nonzero(np.cumsum(f_counts) > i)[0][0]
        x2 = np.nonzero(np.cumsum(f_counts) > j)[0][0]
        
        while x1==x2 and (x1==0 or x1==n-1): #special endpoint case
            #re-sample j and x2
            if x2==0:
                j = np.random.randint(i,a)
            else:
                j = np.random.randint(0,a-i)
            x2 = np.nonzero(np.cumsum(f_counts) > j)[0][0]
        
        #get variable permutation 'jump' size k
        # max_jump = max(1,min(n-1-max(x1,x2),min(x1,x2)))
        
        # if max_jump <= 0: 
        #     print(i,j,x1,x2,max_jump)
        #     print(np.cumsum(f_counts))
        
        k = 1# np.random.randint(1,max_jump+1)
        # if x1+k >= n or x2-k <0:
        #     print(i,j,x1,x2,max_jump,k)
        #     print(np.cumsum(f_counts))
        
        #case for endpoints
        if x2 == 0 or x1 == n-k:
            f_counts[x1] -= 1
            f_counts[x1-k] += 1
            
            f_counts[x2] -= 1
            f_counts[x2+k] +=1
        else:      
            f_counts[x1] -= 1
            f_counts[x1+k] += 1
            
            f_counts[x2] -= 1
            f_counts[x2-k] +=1
        
        return f_counts
    
    '''def mutate_f (self,f):
        #mutate a given f_i
        #add/subtract f from gs
        #mutation method: atom perturbation
        epsilon = 0.005
        newf = np.zeros(self._resolution)
        newf = (1-epsilon)*f + epsilon*self.create_dyad()
        return newf'''
    
    def calc_fitness (self,g_outside,g_outweight):
        #update every member fitness based on outside nodes
        gs=self.get_g()
        fitness = np.zeros(self._popSize)
        for i in range(self._popSize):
            f_i = self._fList[i]
            g_i = (gs - f_i/self._popSize) / (1-1/self._popSize)
            gavg = (g_outside + g_i   )/(1+g_outweight)
            fitness[i] = self.pairwise(gavg,f_i)
        return fitness

    def pairwise (self,g,f):
        #find the payoff of f competing with g
        return sum(f*g.cumsum()-0.5*f*g)

    
    def check_f (self,f):
        #check that f has mean b and sums to 1
        b=self._budget
        if abs(sum(f)-1) > 10e-6:
            print(np.sum(f))
            return False
        if abs(sum(np.arange(self._resolution)*f)-b) > 10e-6:
            print(abs(sum(np.arange(self._resolution)*f)-b))
            return False
        return True
    
    def check_pop(self):
        for f in self._fList:
            t = self.check_f(f)
            if not t:
                return t
        return True

    
    def step(self,g,weight):
        #updates the f_list
        
        
        fitness = self.calc_fitness(g,weight)
        
        #select the f to replace
        #alt methods to try:
        #hard max        
        repl_i = np.argmin(fitness + 10**-6*np.random.rand(len(fitness)))
        # repr_j = np.argmax(fitness + 10**-6*np.random.rand(len(fitness)))
        
        #soft max
        
        #proportial selection
        # prob_replace = 1-fitness
        # prob_replace = prob_replace/sum(prob_replace)
        
        # prob_reproduce = fitness
        # prob_reproduce = prob_reproduce/sum(prob_reproduce)
        
        # repl_i = np.random.choice(range(self._popSize),p=prob_replace)
        # repr_j = np.random.choice(range(self._popSize),p=prob_reproduce)
        
        #scaled proportional
        maxf = max(fitness)
        minf = min(fitness)
        fitness = (fitness-minf)/(maxf-minf + 10**-6) #avoid divide by 0
        
        # prob_replace = 1-fitness
        # prob_replace = prob_replace/sum(prob_replace)
        
        prob_reproduce = fitness
        prob_reproduce = prob_reproduce/sum(prob_reproduce)
        
        #select multiple to reproduce
        # repl_i = np.random.choice(range(self._popSize),p=prob_replace, size=3)
        # repr_j = np.random.choice(range(self._popSize),p=prob_reproduce, size=3)
        # for k in range(3):
        #     self.update_f(repl_i[k],repr_j[k])
        
        repr_j = np.random.choice(range(self._popSize),p=prob_reproduce)
        self.update_f(repl_i, repr_j)
        
        
    def update_f(self,i,j):
        
        old = self._fList[i]
        old_counts = self._f_counts_list[i]
        
        new_counts = self.mutate_f_counts(old_counts)
        new = new_counts/(self._atoms * 1.0)
        
        self._gself += (new-old)/self._popSize
        
        self._fList[i] = new
        self._f_counts_list[i] = new_counts
        
        


class PopNetwork :
    #network class 
    #networkx of populations
    #networkx can manage network atributes
    #each time step, each population must select some 1% to replace, then must recalculate their g and tell others of it
    def __init__ (self, w_matrix, b_list, n, p, r=1000):
        self.P_array = []
        self._numPops = len(w_matrix)
        self._resolution = n
        for i in range(self._numPops):
            self.P_array.append(Population(b_list[i], n, p))
        self._wMat = w_matrix
        self.G = nx.convert_matrix.from_numpy_matrix(w_matrix)
        
        self.record = [self.find_comp_outcome()]
        self.timer = 0
        self.recordrate = r
    
    def calc_outerg (self, ni):
        #calc g_outer and g_outweights for P.calc_fitness for each node
        #each node has weight 1 with self, edge weights are how more/less plays with other node
        n=self._resolution
        neighbors = [j for j in nx.neighbors(self.G,ni)]
        outerg = np.zeros(n)
        totweight = 0
        for j in neighbors:
            jw = self.G[ni][j]['weight']
            outerg += jw*self.P_array[j].get_g()
            totweight += jw
            
        return (outerg, totweight)
            
    def step(self):
        k=self._numPops
        outergs = []
        totweights = []
        for i in range(k):    
            (outerg,totweight) = self.calc_outerg(i)
            outergs.append(outerg)
            totweights.append(totweight)
        
        for i in range(k):
            self.P_array[i].step(outergs[i],totweights[i])
        
        self.timer += 1
        if self.timer%self.recordrate==0:
            self.record.append(self.find_comp_outcome())
        
        
    def plot_pop(self):
        plt.figure()
        for p in self.P_array:
            p.plot_f_sum()
        plt.xlabel('X (budget)')
        plt.ylabel('Density')
        plt.title('Populations at time='+str(self.timer))
        plt.legend([i for i in range(self._numPops)])
            
            
    def find_comp_outcome(self):
        outcomes = -1.0+0*self._wMat
        for e in self.G.edges():
            i,j = e
            Pi_g = self.P_array[i].get_g()
            Pj_g = self.P_array[j].get_g()
            outcomes[i,j] = self.P_array[0].pairwise(Pi_g,Pj_g)
            outcomes[j,i] = 1-outcomes[i,j]
        return outcomes
    
    def check_network(self):
        for p in self.P_array:
            p.check_pop()
            

print("test")
print("Current design: ")
print("1-unit permutation jumpsize, hard min replaced with top proportional fitness")
#P=Population(5,10,5)
#flist = P.get_fList()
#f1=P.get_fList()[0]


# pnetwork.plot_pop()
# #nx.draw(pnetwork.G, with_labels=True, font_weight='bold')
# #plt.show()
# # for t_step in range(100000):
# #     pnetwork.step()
# # pnetwork.plot_pop()

# print("The abc line case 2")
# # w=np.matrix([(0,100,0),(100,0,10),(0,100,0)])
# w=np.matrix([(0,1,0),(1,0,1),(0,1,0)])
# pnetwork=PopNetwork(w, [10,20,30], 100, 5)
# pnetwork.plot_pop()

# #for its in range(100):
# for t_step in range(500000):
#     pnetwork.step()
#     #if (its+1)%10 == 0 :
# pnetwork.plot_pop()

# print("The abcd flag case 1")
# w=np.matrix([(0,1,0,0.5),(1,0,1,0.5),(0,1,0,0.5),(0.5,0.5,0.5,0)])
# pnetwork=PopNetwork(w, [5,15,25,30], 100, 5)
# nx.draw(pnetwork.G, with_labels=True, font_weight='bold')
# pnetwork.plot_pop()

# print("The colonel blotto case 3 (more players)")
# w=np.matrix([(0,1000),(1000,0)])
# pnetwork=PopNetwork(w, [20,40], 100, 10)
# nx.draw(pnetwork.G, with_labels=True, font_weight='bold')
# pnetwork.plot_pop()

# print("The rich case 1")
# w=np.matrix([(0,1000,1),(1000,0,1),(0.001,0.001,0)])
# pnetwork=PopNetwork(w, [10,15,30], 100, 5)
# nx.draw(pnetwork.G, with_labels=True, font_weight='bold')
# pnetwork.plot_pop()

# print("then the colonel blotto case")
# w = np.matrix([(0,10000),(10000,0)])
# pnetwork=PopNetwork(w, [10,20], 100, 5)
# pnetwork.plot_pop()

print("The stacked case 1")
w=np.matrix([(0,1,1),(1,0,1),(1,1,0)])
pnetwork=PopNetwork(w, [20,21,22], 100, 5)
nx.draw(pnetwork.G, with_labels=True, font_weight='bold')
pnetwork.plot_pop()

n_iterations = 2000000

for t_step in range(n_iterations):
    pnetwork.step()
    if t_step%100000==0:
        print(t_step)
pnetwork.plot_pop()


# outcome_m = [pnetwork.record[i][0,1] for i in range(0,5000001,100)]
# outcome_m1 = [pnetwork.record[i][1,0] for i in range(0,5000001,100)]
n_record = len(pnetwork.record)


# with open('colonelcase3_2.csv', 'w', newline='') as csvfile:
#     cwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     for i in range(n_record):
#         # cwriter.writerow([outcome_m[i], outcome_m1[i]])
#         cwriter.writerow([pnetwork.record[i][0,1], pnetwork.record[i][1,0]])
        
# out01 = [pnetwork.record[i][0,1] for i in range(0,n_record)]
# out03 = [pnetwork.record[i][0,3] for i in range(0,n_record)]
# out12 = [pnetwork.record[i][1,2] for i in range(0,n_record)]
# out13 = [pnetwork.record[i][1,3] for i in range(0,n_record)]
# out23 = [pnetwork.record[i][2,3] for i in range(0,n_record)]

with open('stacked1.csv', 'w', newline='') as csvfile:
    cwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(n_record):
        mi = pnetwork.record[i]
        
        cwriter.writerow([mi[0,1], mi[0,2], mi[1,2]])


