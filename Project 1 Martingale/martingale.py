""""""                                        
"""Assess a betting strategy.                                        
                                        
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
                                        
Student Name: Hossein Sahour                                        
GT User ID: hsahour3                                        
GT ID: 903941641                                        
"""                                        
import numpy as np
import matplotlib.pyplot as plt

#credentials                                        
def author():                                                        
    return "hsahour3"                                     
                                        
                                        
def gtid():                                                                     
    return 903941641     

#function to determine win/lose condition                                        
def get_spin_result(win_prob):                                        
    """                                        
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.                                        
                                        
    :param win_prob: The probability of winning                                        
    :type win_prob: float                                        
    :return: The result of the spin.                                        
    :rtype: bool                                        
    """                                        
    result = False                                        
    if np.random.random() <= win_prob:                                        
        result = True                                        
    return result   

#simulation function 
def simulate_bet_sequence(win_prob, num_bets, target_winnings, max_loss=None):
    winnings = np.zeros(num_bets + 1)
    bet_amount = 1

    for i in range(1, num_bets + 1):
        if winnings[i-1] < target_winnings and (max_loss is None or winnings[i-1] > max_loss):
            bet_amount = min(bet_amount, target_winnings - winnings[i-1])
            if max_loss is not None:
                bet_amount = min(bet_amount, -max_loss - winnings[i-1])
            won = get_spin_result(win_prob)
            if won:
                winnings[i] = winnings[i-1] + bet_amount
                bet_amount = 1
            else:
                winnings[i] = winnings[i-1] - bet_amount
                bet_amount *= 2
        else:
            winnings[i] = max(winnings[i-1], max_loss) if max_loss is not None else winnings[i-1]

    return winnings

def plot_figures(all_winnings, title_prefix, fig_start_num):
    means = np.mean(all_winnings, axis=0)
    std_devs = np.std(all_winnings, axis=0, ddof=0)
    medians = np.median(all_winnings, axis=0)

    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    # Mean and standard deviation
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(means, label='Mean Winnings', linewidth=2)
    plt.plot(means + std_devs, 'r', label='Mean + Std Dev', linestyle='dashed')
    plt.plot(means - std_devs, 'g', label='Mean - Std Dev', linestyle='dashed')
    plt.title(f'{title_prefix} Mean Winnings per Bet')
    plt.xlabel('Bet Number')
    plt.ylabel('Winnings')
    plt.ylim(-256, 100)
    plt.xlim(0, 300)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figure_{fig_start_num}.png')

    # Median and standard deviation
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(medians, label='Median Winnings', linewidth=2)
    plt.plot(medians + std_devs, 'r', label='Median + Std Dev', linestyle='dashed')
    plt.plot(medians - std_devs, 'g', label='Median - Std Dev', linestyle='dashed')
    plt.title(f'{title_prefix} Median Winnings per Bet')
    plt.xlabel('Bet Number')
    plt.ylabel('Winnings')
    plt.ylim(-256, 100)
    plt.xlim(0, 300)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figure_{fig_start_num+1}.png')
    
def write_statistics_to_file(filename, all_winnings, title_prefix):
    with open(filename, 'a') as file:
        file.write(f'{title_prefix}\n')
        file.write('Spin Number, Mean Winnings, Median Winnings, Std Deviation\n')
        
        means = np.mean(all_winnings, axis=0)
        medians = np.median(all_winnings, axis=0)
        std_devs = np.std(all_winnings, axis=0, ddof=0)

        for i in range(len(means)):
            file.write(f'{i}, {means[i]:.2f}, {medians[i]:.2f}, {std_devs[i]:.2f}\n')

        file.write('\n')
        
def calculate_success_probability(all_winnings, target_winnings):
    num_successes = np.sum([1 for winnings in all_winnings if winnings[-1] >= target_winnings])
    total_episodes = len(all_winnings)
    return num_successes / total_episodes

def mean_bets_to_reach_goal(all_winnings, goal):
    bet_counts = []
    for winnings in all_winnings:
        for i, amount in enumerate(winnings):
            if amount >= goal:
                bet_counts.append(i)
                break
    return np.mean(bet_counts)


def test_code():
    win_prob = 18/38
    num_episodes = 1000
    num_bets = 1000
    target_winnings = 80
    np.random.seed(gtid())

    # Experiment 1
    all_winnings_exp1 = np.array([simulate_bet_sequence(win_prob, num_bets, target_winnings) for _ in range(num_episodes)])

    # Figure 1: Plot for 10 episodes
    plt.figure(figsize=(10, 6), dpi=100)
    for i in range(10):
        plt.plot(all_winnings_exp1[i], label=f'Episode {i+1}')
        
    plt.title('Experiment 1: Winnings Over Time for 10 Episodes')
    plt.xlabel('Bet Number')
    plt.ylabel('Winnings')
    plt.ylim(-256, 100)
    plt.xlim(0, 300)
    plt.legend()
    plt.savefig('figure_1.png')

    # Figures 2 and 3
    plot_figures(all_winnings_exp1, "Experiment 1:", 2)

    # Experiment 1
    all_winnings_exp1 = np.array([simulate_bet_sequence(win_prob, num_bets, target_winnings) for _ in range(num_episodes)])
    success_probability = calculate_success_probability(all_winnings_exp1, target_winnings)
#    expected_value_exp1 = np.mean([winnings[-1] for winnings in all_winnings_exp1])

    mean_bets_exp1 = mean_bets_to_reach_goal(all_winnings_exp1, target_winnings)
    expected_value_exp1=(target_winnings/mean_bets_exp1)*1000


    # Experiment 2 (Realistic Simulator)
    max_loss = -256
    all_winnings_exp2 = np.array([simulate_bet_sequence(win_prob, num_bets, target_winnings, max_loss) for _ in range(num_episodes)])
    success_probability_exp2 = calculate_success_probability(all_winnings_exp2, target_winnings)
    expected_value_exp2 = np.mean([winnings[-1] for winnings in all_winnings_exp2])

    # Figures 4 and 5
    plot_figures(all_winnings_exp2, "Experiment 2 (Limited Bankroll):", 4)

    # Writing results to file
    with open('p1_results.txt', 'w') as file:
        file.write('Project 1 Simulation Results\n')
        file.write('===========================\n\n')

        file.write('Experiment 1 Results:\n')
        file.write(f'- Probability of winning $80 within 1000 bets: {success_probability:.4f}\n')
        file.write(f'- Estimated expected value of winnings after 1000 bets: ${expected_value_exp1:.2f}\n\n')

        file.write('Experiment 2 Results:\n')
        file.write(f'- Probability of winning $80 within 1000 bets: {success_probability_exp2:.4f}\n')
        file.write(f'- Estimated expected value of winnings after 1000 bets: ${expected_value_exp2:.2f}\n')


if __name__ == "__main__":
    test_code()