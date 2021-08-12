import pandas as pd
import numpy as np
import tqdm
import itertools

def generate_cycle(size:int) -> [[int]]:
    arr = []
    for i in range(size):
        row = []
        for j in range(size):
            if abs(i-j) == 1 or abs(i-j) == size - 1:
                row.append(1)
            else:
                row.append(0)
        arr.append(row)
    return arr

def generate_graph_param(graph_type:str, n:int)->[[int]]:
    if graph_type.lower() == 'c':
        return generate_cycle(n)
    elif graph_type.lower() == 'w':
        return generate_wheel(n)
    elif graph_type.lower() == 's':
        return generate_star(n)
    elif graph_type.lower() == 'q':
        return generate_cube(n)
    elif graph_type.lower() == 'p':
        return generate_complete(n)


    
def build_identity_comp(vertices:int) -> [int]:
    return [[0 if i == j else 1 for i in range(vertices)] for j in range(vertices)]


def product_strat_value(arr1, strat11, strat12, arr2, strat21, strat22) -> (float,int):
    i1 = np.array(build_identity_comp(len(arr1)))
    arr1 = np.array(arr1)
    comp1 = np.subtract(i1,arr1)
    i2 = np.array(build_identity_comp(len(arr2)))
    arr2 = np.array(arr2)
    comp2 = np.subtract(i2,arr2)
    winning_cases = 0
    for q11 in range(len(strat11)):
        for q12 in range(len(strat12)):
            for q21 in range(len(strat21)):
                for q22 in range(len(strat22)):
                    if ((q11 == q12 and strat11[q11]==strat12[q12]) or comp1[strat11[q11]][strat12[q12]])  and (comp2[strat21[q21]][strat22[q22]] or (q11 == q12 and strat11[q11]==strat12[q12])):
                        winning_cases += 1
    return (winning_cases / len(strat11) /len(strat12) / len(strat21) / len(strat22), winning_cases)
    

def product_game_value(graph1:[[int]],graph2:[[int]],set_size1:int, set_size2:int) -> (float,int,int):
    max_value = 0
    numerator = 0
    denominator = set_size1 ** 2 * set_size2 ** 2
    n1 = len(graph1)
    n2 = len(graph2)
    strats_count = 0
    strats = []
    #pbar = tqdm(total=factorial(n1+set_size1-1)/factorial(set_size1)/factorial(n1-1) ** 2*factorial(n2+set_size2-1)/factorial(set_size2)/factorial(n2-1) ** 2)
    for strategy1 in itertools.combinations_with_replacement(range(n1),set_size1):
        for strategy2 in itertools.combinations_with_replacement(range(n2),set_size2):
            #pbar.update(1)
            print(strategy1,strategy2)
            strats_count += 1
            value, num = product_strat_value(graph1,strategy1, strategy1,graph2,strategy2,strategy2)
            if value > max_value:
                max_value = value
                numerator = num
                strats = [strategy1,strategy2]
    #pbar.close()
    print(strats_count)
    return (max_value,strats,numerator,denominator)
    
def generate_different_game_product_csv(graph_type:str, n:int, ks:[int], read_file:str, file_name:str):
    values = pd.read_csv(read_file)
    df = pd.DataFrame(index = ks, columns = ks)
    graph = generate_graph_param(graph_type,n)
    for k in ks:
        vals = []
        for m in ks:
            if k > m:
                vals.append('-')
            else:
                value,strats, num, denom = product_game_value(graph, graph, k, m)
                print(k,m,strats)
                vals.append([value, values.loc[k-1][n] * values.loc[m-1][n]])
        df.loc[k] = vals
        df.to_csv(file_name)  
'''*****************************************************************************************************************
*****************************************************************************************************************
*****************************************************************************************************************
******************************************************************************************************************
****************************************************************************************************************'''
def new_product_strat_value(arr1, strat11, strat12, arr2, strat21, strat22) -> (float,int):
    i1 = np.array(build_identity_comp(len(arr1)))
    arr1 = np.array(arr1)
    comp1 = np.subtract(i1,arr1)
    i2 = np.array(build_identity_comp(len(arr2)))
    arr2 = np.array(arr2)
    comp2 = np.subtract(i2,arr2)
    winning_cases = 0
    for q11 in range(len(strat11)):
        for q12 in range(len(strat12)):
            for q21 in range(len(strat21)):
                for q22 in range(len(strat22)):
                    if ((q11 == q12 and strat11[q11]==strat12[q12]) or comp1[strat11[q11]][strat12[q12]])  and (comp2[strat21[q21]][strat22[q22]] or (q11 == q12 and strat11[q11]==strat12[q12])):
                        winning_cases += 1
    return (winning_cases / len(strat11) /len(strat12) / len(strat21) / len(strat22), winning_cases)
    

def new_product_game_value(graph1:[[int]],graph2:[[int]],set_size1:int, set_size2:int) -> (float,int,int):
    max_value = 0
    numerator = 0
    denominator = set_size1 ** 2 * set_size2 ** 2
    n1 = len(graph1)
    n2 = len(graph2)
    strats_count = 0
    strats = []
    #pbar = tqdm(total=factorial(n1+set_size1-1)/factorial(set_size1)/factorial(n1-1) ** 2*factorial(n2+set_size2-1)/factorial(set_size2)/factorial(n2-1) ** 2)
    for strategy1 in itertools.combinations_with_replacement(range(n1),set_size1):
        for strategy2 in itertools.combinations_with_replacement(range(n2),set_size2):
            #pbar.update(1)
            print(strategy1,strategy2)
            strats_count += 1
            value, num = product_strat_value(graph1,strategy1, strategy1,graph2,strategy2,strategy2)
            if value > max_value:
                max_value = value
                numerator = num
                strats = [strategy1,strategy2]
    #pbar.close()
    print(strats_count)
    return (max_value,strats,numerator,denominator)
    
def new_generate_different_game_product_csv(graph_type:str, n:int, ks:[int], read_file:str, file_name:str):
    values = pd.read_csv(read_file)
    df = pd.DataFrame(index = ks, columns = ks)
    graph = generate_graph_param(graph_type,n)
    for k in ks:
        vals = []
        for m in ks:
            if k > m:
                vals.append('-')
            else:
                value,strats, num, denom = product_game_value(graph, graph, k, m)
                print(k,m,strats)
                vals.append([value, values.loc[k-1][n] * values.loc[m-1][n]])
        df.loc[k] = vals
        df.to_csv(file_name)  
 

def product_game_strat_hell(n:int, k:int, m:int):
    graph = generate_cycle(n)
    print(graph)
    i = np.array(build_identity_comp(len(graph)))
    graph = np.array(graph)
    comp = np.subtract(i,graph)
    print(comp)
    strat = list(itertools.product(itertools.product([[0,2]],repeat=k),repeat=m))[0]
    pbar = tqdm.tqdm(total=(n ** (2*k*m)))  
    max_value = 0
    best_strat = []
    for strat in itertools.product(itertools.product(itertools.product(range(n),repeat=2),repeat=k),repeat=m):
        wins = 0
        values = set([])
        pbar.update(1)
        for q_set in itertools.product(itertools.product(range(k), range(m)),repeat=2):
            qk1 = q_set[0][0]
            qm1 = q_set[0][1]
            qk2 = q_set[1][0]
            qm2 = q_set[1][1]
            if (qk1 == qk2 or comp[strat[qm1][qk1][0]][strat[qm2][qk2][0]]) and (qm1 == qm2 or comp[strat[qm1][qk1][1]][strat[qm2][qk2][1]]):
                wins += 1
        value = wins/(k*k*m*m)
        if value > max_value:
            max_value = value
            best_strat = strat    
    pbar.close()
    print(values)
    print("Game Value:", max_value, str(wins),"/", str(k*k*m*m), "Best Strat:",best_strat)


product_game_strat_hell(5,3,3)
#generate_different_game_product_csv('c', 5, [2], "results20.csv", "productgamesonC5b.csv")
'''
def old_strat_value(arr:[[int]], strategy:[int]) -> (float,int):
    i = np.array(build_identity_comp(len(arr)))
    arr = np.array(arr)
    complement = np.subtract(i,arr)
    winning_cases = 0
    for q1 in range(len(strategy)):
        for q2 in range(len(strategy)):
            if q1==q2:
                winning_cases += 1
            elif q1!=q2:
                winning_cases += complement[strategy[q1]][strategy[q2]]
    return (winning_cases / len(strategy) ** 2, winning_cases)

def old_game_value(graph:[[int]], set_size:int)->([int],float,int,int):
    #strategies = create_strategy_list(chain_len,set_size)
    df = pd.DataFrame(columns=["Strategy","Value"])
    max_value = 0
    numerator = 0
    denominator = set_size ** 2
    best_strats = []
    n = len(graph)
    print("Finding value for dimension", n, ", set size", set_size)
    #total_strategies = factorial(n+set_size-1)//factorial(set_size)//factorial(n-1)
    #pbar = tqdm(total=total_strategies)
    for strategy in itertools.product(range(n),repeat=set_size):
        #pbar.update(1)
        value, num = old_strat_value(graph, strategy)
        if value > max_value:
            max_value = value
            numerator = num
            best_strats = [[strategy,strategy]]
        elif value == max_value:
            best_strats.append([strategy,strategy])
        df.append({'Strategy':strategy,'Value':value},ignore_index=True)
    #pbar.close()
    df.to_csv("PermutationCheck.csv")
    return (best_strats, max_value,numerator,denominator)

old_game_value(generate_cycle(5),3)'''