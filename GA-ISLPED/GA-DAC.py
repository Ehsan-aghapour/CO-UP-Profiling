graphs=["YOLOv3","MobileV1"]
target_acc=66
target_graph="YOLOv3"
p_dir='../Profiling'
#p_dir='/home/ehsan/UvA/ARMCL/Rock-Pi/CO-UP-Profiling/Profiling'
#p_dir='/home/ehsan/UvA/ARMCL/Rock-Pi/LW-ARM-CO-UP/New/Model/test'
NLayers={"YOLOv3":75, "MobileV1":14}
model_names = { "MobileV1":"Mobile.h5", "YOLOv3":"YOLOv3.h5" }
target_accuracies={"YOLOv3":[66], "MobileV1":[]}

import sys
import time
import concurrent.futures
sys.path.append(p_dir)
import predict_cost as P
import utils
import pymoo
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
import sys
from tensorflow.keras import layers, models
import pandas as pd
# +
model=None
'''if sys.argv[1]=="y":
    target_graph="YOLOv3"
if sys.argv[1]=="m":
    target_graph="MobileV1"
target_acc=float(sys.argv[2])'''

model_name=model_names[target_graph]
model=models.load_model(model_name)


print(f'Running Ga for model:{target_graph} for target accuracy:{target_acc}')


# +
########################################################
# N with just min freq
def decode_gene_0(v):
    if v==0:
        return "N",[v]
    elif v<6 :
        return "G",[v-1,7]
    elif v<14:
        return "B",[v-6]
    elif v<20:
        return "L",[v-14]
def decoder_0(chromosome):
    freqs=[]
    ps=''
    for gene in chromosome:
        p,fs=decode_gene_2(gene)
        ps+=p
        freqs.append(fs)
    return freqs,ps




#####################################################
# N with max freq and Reorder based on their pareto  
def decode_gene_1(v):
    if v==0:
        return "N",[7]
    elif v<7 :
        return "L",[v-1]
    elif v<15:
        return "B",[v-7]
    elif v<20:
        return "G",[v-15,7]
def decoder_1(chromosome):
    freqs=[]
    ps=''
    for gene in chromosome:
        p,fs=decode_gene_1(gene)
        ps+=p
        freqs.append(fs)
    return freqs,ps

#######################################################
    
# N with 8 B freqs   
def decode_gene_2(v):
    if v<8:
        return "N",[v]
    elif v<14 :
        return "L",[v-8]
    elif v<22:
        return "B",[v-14]
    elif v<27:
        return "G",[v-22,7]


    
def decoder_2(chromosome):
    freqs=[]
    ps=''
    for gene in chromosome:
        p,fs=decode_gene_2(gene)
        ps+=p
        freqs.append(fs)
    return freqs,ps


##############################################################

#just max and min DVFS
translate={0:('N',[7]),
          1:('L',[5]),
          2:('B',[7]),
          3:('G',[4,7]),
          4:('N',[0]),
          5:('L',[0]),
          6:('B',[0]),
          7:('G',[0,0])}
def decode_gene_3(v):
    return translate[v]

def decoder_3(chromosome):
    freqs=[]
    ps=''
    for gene in chromosome:
        p,fs=decode_gene_3(gene)
        ps+=p
        freqs.append(fs)
    return freqs,ps


# -

#np.set_printoptions(threshold=np.inf)
#class MyProblem(ElementwiseProblem):
class MyProblem_0(Problem):
    generation=0
    def set_target_accuracy(self,target_accuracy):
        self.target_accuracy=target_accuracy
        print(f'target accuracy is set to {self.target_accuracy}')
    def __init__(self,_graph,target_accuracy):
        self.target_accuracy=target_accuracy
        self.g=_graph
        self.n=NLayers[_graph]
        print("Initialize the problem for graph with " + str(self.n) + " layers.")
        _xl=np.full(self.n,0)
        _xu=np.full(self.n,20)
        super().__init__(n_var=self.n,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array(_xl),
                         xu=np.array(_xu),
                         vtype=int
                        )
        #self.integer = np.arange(20)

    def worker_function(self,config):
        return P.Inference_Cost(_graph=self.g, _freq=config[0], _order=config[1], _dvfs_delay='variable',_debug=False)

        
    def _evaluate(self, X, out, *args, **kwargs):
        #print(f'target accuracy is {self.target_accuracy}')
        X = np.round(X).astype(int)
        np.set_printoptions(threshold=np.inf)
        #print(X)
        configs=[decoder_0(x1) for x1 in X]
        
        inference_time = np.zeros(X.shape[0])
        avg_power = np.zeros(X.shape[0])
        
        '''start_time = time.time()
        # Iterate over each solution for the second function
        for i,config in enumerate(configs):
            print(f'graph is {self.g} config is {config}')
            inference_time[i],avg_power[i],_ = P.Inference_Cost(_graph=self.g,_freq=config[0],_order=config[1],_dvfs_delay='variable')
            print(inference_time[i],avg_power[i])
            if np.isnan(inference_time[i]):
                print(X[i])
                print(config)
                input("nan")
        end_time = time.time()
        print("Execution time of one call:", end_time - start_time)
        print(inference_time,avg_power)
        inference_time = np.zeros(X.shape[0])
        avg_power = np.zeros(X.shape[0])'''
                
        
            
        # Run in parallel
        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self.worker_function, configs)

        # Iterate through the results and assign them
        for i, result in enumerate(results):
            inference_time[i], avg_power[i], _ = result
            #print(configs[i][1],result)
            if np.isnan(inference_time[i]) or np.isnan(avg_power[i]):
                print(configs[i],result)
                input("nan value detected\n")
            
        end_time = time.time()
        print(f"Gen:{self.generation} Execution time of one call:{end_time - start_time}")
        self.generation=self.generation+1;
        #print(inference_time,avg_power)   
        #input()
            
        x_quantization=np.where(X==0,1,0)
        predicted_accuracy = model.predict(x_quantization).flatten()
        #print(predicted_accuracy)
        #G= predicted_accuracy - self.target_accuracy
        G= self.target_accuracy - predicted_accuracy
        #print(G)
        #input()
        #print(f'time:{inference_time}')
        #print(f'power:{avg_power}')

        out["F"] = [inference_time, avg_power]
        out["G"] = [G]


#np.set_printoptions(threshold=np.inf)
#class MyProblem(ElementwiseProblem):
class MyProblem_1(Problem):
    generation=0
    def set_target_accuracy(self,target_accuracy):
        self.target_accuracy=target_accuracy
        print(f'target accuracy is set to {self.target_accuracy}')
    def __init__(self,_graph,target_accuracy):
        self.target_accuracy=target_accuracy
        self.g=_graph
        self.n=NLayers[_graph]
        print("Initialize the problem for graph with " + str(self.n) + " layers.")
        _xl=np.full(self.n,0)
        _xu=np.full(self.n,20)
        super().__init__(n_var=self.n,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array(_xl),
                         xu=np.array(_xu),
                         vtype=int
                        )
        #self.integer = np.arange(20)

    def worker_function(self,config):
        return P.Inference_Cost(_graph=self.g, _freq=config[0], _order=config[1], _dvfs_delay='variable',_debug=False)

        
    def _evaluate(self, X, out, *args, **kwargs):
        #print(f'target accuracy is {self.target_accuracy}')
        X = np.round(X).astype(int)
        np.set_printoptions(threshold=np.inf)
        #print(X)
        configs=[decoder_1(x1) for x1 in X]
        
        inference_time = np.zeros(X.shape[0])
        avg_power = np.zeros(X.shape[0])
        
        '''start_time = time.time()
        # Iterate over each solution for the second function
        for i,config in enumerate(configs):
            print(f'graph is {self.g} config is {config}')
            inference_time[i],avg_power[i],_ = P.Inference_Cost(_graph=self.g,_freq=config[0],_order=config[1],_dvfs_delay='variable')
            print(inference_time[i],avg_power[i])
            if np.isnan(inference_time[i]):
                print(X[i])
                print(config)
                input("nan")
        end_time = time.time()
        print("Execution time of one call:", end_time - start_time)
        print(inference_time,avg_power)
        inference_time = np.zeros(X.shape[0])
        avg_power = np.zeros(X.shape[0])'''
                
        
            
        # Run in parallel
        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self.worker_function, configs)

        # Iterate through the results and assign them
        for i, result in enumerate(results):
            inference_time[i], avg_power[i], _ = result
            #print(configs[i][1],result)
            if np.isnan(inference_time[i]) or np.isnan(avg_power[i]):
                print(configs[i],result)
                input("nan value detected\n")
            
        end_time = time.time()
        print(f"Gen:{self.generation} Execution time of one call:{end_time - start_time}")
        self.generation=self.generation+1;
        #print(inference_time,avg_power)   
        #input()
            
        x_quantization=np.where(X==0,1,0)
        predicted_accuracy = model.predict(x_quantization).flatten()
        #print(predicted_accuracy)
        #G= predicted_accuracy - self.target_accuracy
        G= self.target_accuracy - predicted_accuracy
        #print(G)
        #input()
        #print(f'time:{inference_time}')
        #print(f'power:{avg_power}')

        out["F"] = [inference_time, avg_power]
        out["G"] = [G]


#np.set_printoptions(threshold=np.inf)
#class MyProblem(ElementwiseProblem):
class MyProblem_2(Problem):
    generation=0
    def set_target_accuracy(self,target_accuracy):
        self.target_accuracy=target_accuracy
        print(f'target accuracy is set to {self.target_accuracy}')
    def __init__(self,_graph,target_accuracy):
        self.target_accuracy=target_accuracy
        self.g=_graph
        self.n=NLayers[_graph]
        print("Initialize the problem for graph with " + str(self.n) + " layers.")
        _xl=np.full(self.n,0)
        _xu=np.full(self.n,26)
        super().__init__(n_var=self.n,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array(_xl),
                         xu=np.array(_xu),
                         vtype=int
                        )
        #self.integer = np.arange(20)

    def worker_function(self,config):
        return P.Inference_Cost(_graph=self.g, _freq=config[0], _order=config[1], _dvfs_delay='variable',_debug=False)

        
    def _evaluate(self, X, out, *args, **kwargs):
        #print(f'target accuracy is {self.target_accuracy}')
        X = np.round(X).astype(int)
        np.set_printoptions(threshold=np.inf)
        #print(X)
        configs=[decoder_2(x1) for x1 in X]
        
        inference_time = np.zeros(X.shape[0])
        avg_power = np.zeros(X.shape[0])
        
        '''start_time = time.time()
        # Iterate over each solution for the second function
        for i,config in enumerate(configs):
            print(f'graph is {self.g} config is {config}')
            inference_time[i],avg_power[i],_ = P.Inference_Cost(_graph=self.g,_freq=config[0],_order=config[1],_dvfs_delay='variable')
            print(inference_time[i],avg_power[i])
            if np.isnan(inference_time[i]):
                print(X[i])
                print(config)
                input("nan")
        end_time = time.time()
        print("Execution time of one call:", end_time - start_time)
        print(inference_time,avg_power)
        inference_time = np.zeros(X.shape[0])
        avg_power = np.zeros(X.shape[0])'''
                
        
            
        # Run in parallel
        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self.worker_function, configs)

        # Iterate through the results and assign them
        for i, result in enumerate(results):
            inference_time[i], avg_power[i], _ = result
            #print(configs[i][1],result)
            if np.isnan(inference_time[i]) or np.isnan(avg_power[i]):
                print(configs[i],result)
                input("nan value detected\n")
            
        end_time = time.time()
        print(f"Gen:{self.generation} Execution time of one call:{end_time - start_time}")
        self.generation=self.generation+1;
        #print(inference_time,avg_power)   
        #input()
            
        x_quantization=np.where(X==0,1,0)
        predicted_accuracy = model.predict(x_quantization).flatten()
        #print(predicted_accuracy)
        #G= predicted_accuracy - self.target_accuracy
        G= self.target_accuracy - predicted_accuracy
        #print(G)
        #input()
        #print(f'time:{inference_time}')
        #print(f'power:{avg_power}')

        out["F"] = [inference_time, avg_power]
        out["G"] = [G]


#np.set_printoptions(threshold=np.inf)
#class MyProblem(ElementwiseProblem):
class MyProblem_3(Problem):
    generation=0
    def set_target_accuracy(self,target_accuracy):
        self.target_accuracy=target_accuracy
        print(f'target accuracy is set to {self.target_accuracy}')
    def __init__(self,_graph,target_accuracy):
        self.target_accuracy=target_accuracy
        self.g=_graph
        self.n=NLayers[_graph]
        print("Initialize the problem for graph with " + str(self.n) + " layers.")
        _xl=np.full(self.n,0)
        _xu=np.full(self.n,7)
        super().__init__(n_var=self.n,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array(_xl),
                         xu=np.array(_xu),
                         vtype=int
                        )
        #self.integer = np.arange(20)

    def worker_function(self,config):
        return P.Inference_Cost(_graph=self.g, _freq=config[0], _order=config[1], _dvfs_delay='variable',_debug=False)

        
    def _evaluate(self, X, out, *args, **kwargs):
        #print(f'target accuracy is {self.target_accuracy}')
        X = np.round(X).astype(int)
        np.set_printoptions(threshold=np.inf)
        #print(X)
        configs=[decoder_3(x1) for x1 in X]
        
        inference_time = np.zeros(X.shape[0])
        avg_power = np.zeros(X.shape[0])
        
        '''start_time = time.time()
        # Iterate over each solution for the second function
        for i,config in enumerate(configs):
            print(f'graph is {self.g} config is {config}')
            inference_time[i],avg_power[i],_ = P.Inference_Cost(_graph=self.g,_freq=config[0],_order=config[1],_dvfs_delay='variable')
            print(inference_time[i],avg_power[i])
            if np.isnan(inference_time[i]):
                print(X[i])
                print(config)
                input("nan")
        end_time = time.time()
        print("Execution time of one call:", end_time - start_time)
        print(inference_time,avg_power)
        inference_time = np.zeros(X.shape[0])
        avg_power = np.zeros(X.shape[0])'''
                
        
            
        # Run in parallel
        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self.worker_function, configs)

        # Iterate through the results and assign them
        for i, result in enumerate(results):
            inference_time[i], avg_power[i], _ = result
            #print(configs[i][1],result)
            if np.isnan(inference_time[i]) or np.isnan(avg_power[i]):
                print(configs[i],result)
                input("nan value detected\n")
            
        end_time = time.time()
        print(f"Gen:{self.generation} Execution time of one call:{end_time - start_time}")
        self.generation=self.generation+1;
        #print(inference_time,avg_power)   
        #input()
            
        x_quantization=np.where(X==0,1,0)
        predicted_accuracy = model.predict(x_quantization).flatten()
        #print(predicted_accuracy)
        #G= predicted_accuracy - self.target_accuracy
        G= self.target_accuracy - predicted_accuracy
        #print(G)
        #input()
        #print(f'time:{inference_time}')
        #print(f'power:{avg_power}')

        out["F"] = [inference_time, avg_power]
        out["G"] = [G]

import random
def define_initial_population(decoder_type=3):
    
    NPU=[]
    L=[]
    if decoder_type in [0,1]:
        NPU=[0]
        L=[1,7]
        up=20
    
    if decoder_type in [2]:
        NPU=[0,7]
        L=[8,13]
        up=26
    
    if decoder_type in [3]:
        NPU=[0,4]
        L=[1,5] 
        up=7
        
    initial_population = np.random.randint(low=0, high=up, size=(200, 75))
    
   
    j=0
    for i in range(1,76):
        arr=np.zeros(75)
        # to select between min freq or max freq (of NPU and L) together
        _index=random.choice([0,-1])
        #6
        arr[:]=L[_index]
        #0
        arr[:i]=NPU[_index]
        #print(arr)
        initial_population[j]=arr
        j=j+1
    for i in range(1,76):
        arr=np.zeros(75)
        _index=random.choice([0,-1])
        #6
        arr[:]=L[_index]
        #0
        arr[i:]=NPU[_index]
        #print(arr)
        initial_population[j]=arr
        j=j+1

    for k,x in enumerate(initial_population):
        print(k,x)
    return initial_population
initial_population=define_initial_population()
#initial_population

# +
def my_callback(algorithm):
    generation = algorithm.n_gen
    population = algorithm.pop.get("X")
    print(f"Generation {generation}:")
    print("Population:", population)

def print_best_objectives(algorithm):
    best_f1 = algorithm.pop.get("F")[:, 0].min()
    best_f2 = algorithm.pop.get("F")[:, 1].min()
    print(f"\nGeneration:{algorithm.n_gen} Best f1: {best_f1}, Best f2: {best_f2}\n******************************************************\n\n\n")    
    if algorithm.n_gen==1:
        # Access the current population
        population = algorithm.pop.get("X")
        # Modify the population (for demonstration purposes, adding random noise)
        population[:len(initial_population)] = initial_population[:]
        # Update the population in the algorithm object
        algorithm.pop.set("X", population)

    
    


# +
problem_2 = MyProblem_2(target_graph,66)
problem_3 = MyProblem_3(target_graph,66)
algorithm = NSGA2(
    pop_size=200,
    eliminate_duplicates=True,
    #init_pop=initial_population,
)    
    
algorithm.callback = my_callback

# Configure the optimization algorithm with the custom callback function
algorithm.callback = print_best_objectives
# Configure the optimization algorithm with the callback function
#algorithm.callback = my_callback
# -

def print_res(res):
    for i in range(len(res.X)):
        if res.F[i][0]!=np.nan:
            x = res.X[i]
            y = res.F[i]
            print(f"Solution {i+1}: Decision Variables , Objective Values = {y}")


# +
def plot_res(res):
    plot = Scatter()
    plot.add(res.F, edgecolor="red", facecolor="none")
    plot.save(f"{target_graph}_{target_acc}.jpg")
    plot.show()
    
#plot_res()


# -

def to_csv(res):
    X = np.round(res.X).astype(int)

    configs = [decoder(x1) for x1 in X]

    data = pd.DataFrame({
        'graph': [target_graph] * len(configs),
        'order': [config[1] for config in configs],
        'freq': [tuple(tuple(c) for c in config[0]) for config in configs],
        'pred_average_power': res.F[:, 1],
        'pred_total_time': res.F[:, 0],  # Assuming the first objective is objective_1
    })
    #display(data)
    print(f'writing results df to {target_graph}_{target_acc}.csv')
    data.to_csv(f"{target_graph}_{target_acc}.csv")



# +
def run(n=400,_target_acc=66,_problem=problem_2):
    global res,target_acc
    _problem.set_target_accuracy(_target_acc)
    target_acc=_target_acc
    res = minimize(_problem,
                   algorithm,
                   ("n_gen", n),
                   verbose=True,
                   seed=1,
                   save_history=True,
                  )

    import pickle
    with open(f'{target_graph}-{target_acc}.pkl', "wb") as f:
        pickle.dump(res, f)
    return res

run_flag=True
if run_flag==True:
    global initial_population
    target_accuracies = [accuracy / 10 for accuracy in range(647, 688, 2)]
    target_accuracies.append(68.77)
    print(target_accuracies)
    for target in target_accuracies:
        initial_population=define_initial_population(decoder_type=3)
        res=run(n=1000,_target_acc=target,_problem=problem_3)
        plot_res(res)
        to_csv(res)
        initial_population=res.X
        res2=run(n=1000,_target_acc=target,_problem=problem_3)
        plot_res(res2)
        to_csv(res2)
# -

if False:
    fff=[[2], [0], [0], [0], [2, 7], [1, 7], [3, 7], [1], [2], [4], [2], [7], [3, 7], [3], [0, 7], [7], [2], [5], [2, 7], [3, 7], [1], [4], [0], [7], [3], [3], [1, 7], [0, 7], [2, 7], [3], [1, 7], [2], [4], [4], [7], [0], [7], [2], [0], [0], [5], [0], [4, 7], [1], [1, 7], [3], [3], [0], [4, 7], [1, 7], [0], [7], [3, 7], [4, 7], [3], [0, 7], [5], [2, 7], [5], [7], [1, 7], [2], [7], [2], [0, 7], [4], [7], [4], [4], [5], [3], [2, 7], [2, 7], [1], [2]]
    ordd='BLNBGGGBBBBBGLGBBBGGLLBBLLGGGLGBLBBBBLNLLLGLGBLBGGNBGGBGBGBBGBBBGBBBLBLGGLB'
    P.Inference_Cost(_graph='YOLOv3',_order=ordd,_freq=fff,_debug=True)

d2=pd.read_csv(f"{target_graph}_{target_acc}.csv")
d3=utils.format_freqs(d2['freq'])
d4=utils.format_to_list(d2['freq'])
P.Inference_Cost(_graph=d2.iloc[0]['graph'],_freq=d4[0],_order=d2.iloc[0]['order'],_dvfs_delay='variable')

Evals_df=d2
grouped = Evals_df[Evals_df['graph']==target_graph].groupby('order')
grouped.apply(print)
unique_values_order = Evals_df[Evals_df['graph']==target_graph]['order'].unique()
#print(grouped.unique())
print(unique_values_order)

# +
import pygmo as pg
import numpy as np
import pandas as pd
import pickle

'''model_name=model_names[target_graph]
model=models.load_model(model_name)
problem = MyProblem(target_graph,66)'''

'''res = minimize(problem,
               algorithm,
               ("n_gen", 200),
               verbose=False,
               seed=1,
                save_history=True,
              )

import pickle
with open(f'{target_graph}-{target_acc}', "wb") as f:
    pickle.dump(res, f)'''


def hv_cal(Name_GA,Name_BL):
    with open(Name_GA,'rb') as f:
        res=pickle.load(f)

    df = pd.read_csv(Name_BL)  # Replace with your CSV file path
    pareto_frontier_baseline = df[['Time', 'Power']].to_numpy()
    pareto_frontier_ga=res.F
    #display(pareto_frontier_baseline)
    #display(pareto_frontier_ga)
    plot = Scatter()
    plot.add(res.F, edgecolor="red", facecolor="none")
    plot.add(pareto_frontier_baseline, edgecolor="blue", facecolor="none")
    plot.show()
    ref_point = [max(pareto_frontier_ga[:, 0].max(), pareto_frontier_baseline[:, 0].max()) + 1,
                 max(pareto_frontier_ga[:, 1].max(), pareto_frontier_baseline[:, 1].max()) + 1]

    hv_ga = pg.hypervolume(pareto_frontier_ga)
    hv_baseline = pg.hypervolume(pareto_frontier_baseline)

    volume_ga = hv_ga.compute(ref_point)
    #hv_ga.compute([100000,100000])
    volume_baseline = hv_baseline.compute(ref_point)

    print("Hypervolume GA:", volume_ga/1000000)
    print("Hypervolume Baseline:", volume_baseline/1000000)
    d=pd.DataFrame(pareto_frontier_ga,columns=["Time","Power"])
    d.to_csv("motivation_example_islped_paper.csv")
    


_Name_GA="YOLOv3-66.pkl"
_Name_BL='YOLOv3_ParotoFront68.2711766072.csv'
_Name_GA="R/GA/"+_Name_GA
_Name_BL="R/Baseline/"+_Name_BL
hv_cal(Name_GA=_Name_GA,Name_BL=_Name_BL)
# +
def hv_cal_ref(file_name,ref_point):
    df=pd.read_csv(file_name)
    df['pred_total_time']=df['pred_total_time']/1000
    df['pred_average_power']=df['pred_average_power']/1000
    pareto_frontier=df[['pred_total_time','pred_average_power']].to_numpy()
    #print(pareto_frontier)
    #print(pareto_frontier.shape)
    hv=pg.hypervolume(pareto_frontier)
    #print(hv)
    volume=hv.compute(ref_point)
    print(f'file {file_name} hv {volume}')
    return(volume)

#hv_cal_ref(file_name='csvs/YOLOv3_64.7.csv',ref_point=[12.31,5.55])
#max power and time with little cpu
_ref_point=[12.31,5.55]
#_ref_point=[20,10]
import os
# Specify the directory path
directory = 'csvs/'
# Get the list of files in the directory
files = os.listdir(directory)
files = [file for file in files if not file.startswith(".~lock")]
hvs=[]
#df_hv = pd.DataFrame(columns=['Filename', 'HV'])
for f in files:
    print(f'\nCalculating hv for file {f}')
    hv=hv_cal_ref(file_name=directory+f,ref_point=_ref_point)
    hvs.append({'Filename':f,'HV':hv})
    #df_hv = df_hv.append({'Filename': f, 'HV': hv}, ignore_index=True)

df_hv=pd.DataFrame(hvs)
#display(df_hv)
df_hv=df_hv.sort_values(by='HV',ascending=True)
display(df_hv)
df_hv.to_csv('hvs.csv')
# -

if False:
    #inference_time,avg_power,_=P.Inference_Cost(_graph=graph,_freq=config[0],_order=config[1],_dvfs_delay='variable')
    x='[ 8 14  0  6  3  2  4  7  8 10  8 13  4 17]'
    x='[ 2  7 16 11  1  5  5  7 18  4  9  4  8 13 16  6  9  6  7 18  2 13 13  4\
       5  1 14 14  1  3  1 13 16  9 18 12  1 15  3  3 11  4  6  1  9 10  2  3\
      10  6  1  5 10  8  7  2  1  1  3  9  0  6 10 14 16 15  8  9 16  3 10  5\
      12 17 18]'
    # Remove the square brackets and split the string by spaces
    values = x.strip('[]').split()

    # Convert the values to integers
    x = np.array([int(value) for value in values])
    config=decoder(x)
    inference_time,avg_power,_=P.Inference_Cost(_debug=False,_graph=graph,_freq=config[0],_order=config[1],_dvfs_delay='variable')
    x=np.where(x==0,1,0)
    print(x)
    #model.predict(x.reshape(1,NLayers[graph]))
    print(inference_time,avg_power)
    import math
    np.isnan(inference_time)

x=np.zeros(75)
x[26:]=1
x_quantization=np.array([x])
x_quantization

model.predict(x_quantization).flatten()
