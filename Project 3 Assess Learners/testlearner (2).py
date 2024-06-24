""""""  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	   			  		 			     			  	 
All Rights Reserved  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	   			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	   			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	   			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   			  		 			     			  	 
or edited.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	   			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	   			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   			  		 			     			  	 
GT honor code violation.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	   			  		 			     			  	 
"""  		

import numpy as np
import matplotlib.pyplot as plt
import time
from DTLearner import DTLearner
from RTLearner import RTLearner
from BagLearner import BagLearner
from InsaneLearner import InsaneLearner  
import sys


def load_data(file_path):
    np.random.seed(903941641) 
    data = np.genfromtxt(file_path, delimiter=",", skip_header=1, usecols=range(1, 10))
    np.random.shuffle(data)
    X, Y = data[:, :-1], data[:, -1]
    split_index = int(0.6 * len(Y))
    return X[:split_index], Y[:split_index], X[split_index:], Y[split_index:]


def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def plot_rmse(leaf_sizes, rmse_in, rmse_out, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, rmse_in, label='In-sample RMSE')
    plt.plot(leaf_sizes, rmse_out, label='Out-of-sample RMSE', linestyle='--')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_comparison(metrics, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    for key, values in metrics.items():
        plt.plot(values, label=key)
    plt.xlabel('Training Size')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def experiment_1(X_train, Y_train, X_test, Y_test):
    leaf_sizes = np.arange(1, 51)
    rmse_in, rmse_out = [], []
    for leaf_size in leaf_sizes:
        learner = DTLearner(leaf_size=leaf_size)
        learner.add_evidence(X_train, Y_train)
        rmse_in.append(compute_rmse(Y_train, learner.query(X_train)))
        rmse_out.append(compute_rmse(Y_test, learner.query(X_test)))
    plot_rmse(leaf_sizes, rmse_in, rmse_out, 'DTLearner: In-sample vs Out-of-sample RMSE', 'exp1_dt_rmse.png')

def experiment_2(X_train, Y_train, X_test, Y_test):
    leaf_sizes = np.arange(1, 51)
    rmse_in, rmse_out = [], []
    for leaf_size in leaf_sizes:
        learner = BagLearner(lambda: DTLearner(leaf_size=leaf_size), {}, 20)
        learner.add_evidence(X_train, Y_train)
        rmse_in.append(compute_rmse(Y_train, learner.query(X_train)))
        rmse_out.append(compute_rmse(Y_test, learner.query(X_test)))
    plot_rmse(leaf_sizes, rmse_in, rmse_out, 'Bagged DTLearner: In-sample vs Out-of-sample RMSE', 'exp2_bag_dt_rmse.png')

def experiment_3(X_train, Y_train, X_test, Y_test):
    leaf_sizes = np.arange(1, 51) 
    metrics = {'DTLearner': [], 'RTLearner': [], 'Bagged DTLearner': [], 'Bagged RTLearner': []}
    r2_scores = {'DTLearner': [], 'RTLearner': [], 'Bagged DTLearner': [], 'Bagged RTLearner': []}  
    
    for leaf_size in leaf_sizes:
        learners = {
            'DTLearner': DTLearner(leaf_size=leaf_size),
            'RTLearner': RTLearner(leaf_size=leaf_size),
            'Bagged DTLearner': BagLearner(lambda: DTLearner(leaf_size=leaf_size), {}, 20),
            'Bagged RTLearner': BagLearner(lambda: RTLearner(leaf_size=leaf_size), {}, 20),
            #'InsaneLearner': InsaneLearner(verbose=False)
        }

        for name, learner in learners.items():
            learner.add_evidence(X_train, Y_train)
            pred_Y = learner.query(X_test)
            mae = compute_mae(Y_test, pred_Y)

            y_mean = np.mean(Y_test)
            ss_tot = np.sum((Y_test - y_mean) ** 2)
            ss_res = np.sum((Y_test - pred_Y) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            metrics[name].append(mae)
            r2_scores[name].append(r2)  

    plt.figure(figsize=(10, 6))
    for learner_name, mae_values in metrics.items():
        plt.plot(leaf_sizes, mae_values, label=f'{learner_name} MAE')
    plt.xlabel('Leaf Size')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Learner Comparison on MAE by Leaf Size')
    plt.legend()
    plt.savefig('exp3_mae_comparison_by_leaf_size.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for learner_name, r2_values in r2_scores.items():
        plt.plot(leaf_sizes, r2_values, label=f'{learner_name} $R^2$')
    plt.xlabel('Leaf Size')
    plt.ylabel('$R^2$ Score')
    plt.title('Learner Comparison on $R^2$ by Leaf Size')
    plt.legend()
    plt.savefig('exp3_r2_comparison_by_leaf_size.png')
    plt.close()



def experiment_3_2(X_train, Y_train, X_test, Y_test, chunk_sizes):
    metrics = {'DTLearner': [], 'RTLearner': []}
    times = {'DTLearner': [], 'RTLearner': []}
    
    for chunk_size in chunk_sizes:
        current_chunk_size = min(chunk_size, X_train.shape[0])
        X_train_chunk = X_train[:current_chunk_size]
        Y_train_chunk = Y_train[:current_chunk_size]

        learners = {
            'DTLearner': DTLearner(leaf_size=1),  
            'RTLearner': RTLearner(leaf_size=1),
        }

        for name, learner in learners.items():
            start_time = time.time()
            learner.add_evidence(X_train_chunk, Y_train_chunk)
            training_time = time.time() - start_time

            times[name].append(training_time)

    plt.figure(figsize=(10, 6))
    for learner_name, time_values in times.items():
        plt.plot(chunk_sizes, time_values, label=f'{learner_name} Training Time')
    plt.xlabel('Training Chunk Size')
    plt.ylabel('Training Time (seconds)')
    plt.title('Learner Comparison on Training Time by Training Chunk Size')
    plt.legend()
    plt.savefig('exp3_2_training_time_comparison.png')
    plt.close()


def main():
    if len(sys.argv) != 2:  
        print("Usage: python testlearner.py <path_to_dataset>")
        sys.exit(1)
    X_train, Y_train, X_test, Y_test = load_data(sys.argv[1])
    
    experiment_1(X_train, Y_train, X_test, Y_test)
    experiment_2(X_train, Y_train, X_test, Y_test)
    experiment_3(X_train, Y_train, X_test, Y_test)
    
    chunk_sizes = [50, 100, 200, 300, 400, 500]
    experiment_3_2(X_train, Y_train, X_test, Y_test, chunk_sizes)

if __name__ == "__main__":
    main()
