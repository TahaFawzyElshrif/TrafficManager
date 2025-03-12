from tabulate import tabulate
from colorama import Fore, Style

def show_ppo_progress(iteration_index, result, show_each_agent=True):
    
    print(Fore.BLACK + "______________________________________" + Style.RESET_ALL)
    print(Fore.MAGENTA + f"ITERATION {iteration_index}" + Style.RESET_ALL)
    print(Fore.BLACK + "______________________________________" + Style.RESET_ALL)

    cum_loss = 0
    cum_reward = 0
    count = 0

    # Get policy rewards safely
    policy_rewards = result.get('env_runners', {}).get('policy_reward_mean', {})

    if not policy_rewards:
        print(Fore.RED + "TERMINATED EPISODE, NO REWARD" + Style.RESET_ALL)

    headers = ["Agent", "Total Loss", "Policy Loss", "Value Function Loss", "Reward"]
    table_data = []

    # Loop through all agents in the learner info
    for agent_id, stats in result.get('info', {}).get('learner', {}).items():
        learner_stats = stats.get('learner_stats', {})  # Prevent KeyError
        
        total_loss = learner_stats.get('total_loss', 0)
        policy_loss = learner_stats.get('policy_loss', 0)
        vf_loss = learner_stats.get('vf_loss', 0)
        
        # Get reward safely
        reward = policy_rewards.get(agent_id, 0)

        cum_reward += reward
        cum_loss += total_loss
        count += 1
        if show_each_agent:
            table_data.append([
                agent_id,
                f"{Fore.RED}{total_loss}{Style.RESET_ALL}",
                f"{Fore.YELLOW}{policy_loss}{Style.RESET_ALL}",
                f"{Fore.GREEN}{vf_loss}{Style.RESET_ALL}",
                f"{Fore.CYAN}{reward}{Style.RESET_ALL}"
            ])

    # Print the table if enabled
    if show_each_agent:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    print(Fore.MAGENTA + f"ITERATION {iteration_index} FINISHED, with Reward {cum_reward}, Loss {cum_loss}" + Style.RESET_ALL)
    print(Fore.BLACK + "______________________________________" + Style.RESET_ALL)

    return cum_loss/count, cum_reward/count # Return average loss and reward


def show_dqn_progress(iteration_index, result, show_each_agent=True):
    
    print(Fore.BLACK + "______________________________________" + Style.RESET_ALL)
    print(Fore.MAGENTA + f"ITERATION {iteration_index}" + Style.RESET_ALL)
    print(Fore.BLACK + "______________________________________" + Style.RESET_ALL)

    cum_loss = 0
    cum_reward = 0
    count = 0

    # Get policy rewards safely
    policy_rewards = result.get('env_runners', {}).get('policy_reward_mean', {})

    if not policy_rewards:
        print(Fore.RED + "TERMINATED EPISODE, NO REWARD" + Style.RESET_ALL)

    headers = ["Agent", "Total Loss", "Policy Loss", "Value Function Loss", "Reward"]
    table_data = []

    # Loop through all agents in the learner info
    for agent_id, stats in result.get('info', {}).get('learner', {}).items():
        learner_stats = stats.get('learner_stats', {})  # Prevent KeyError
        
        total_loss = learner_stats.get('total_loss', 0)
        policy_loss = learner_stats.get('policy_loss', 0)
        vf_loss = learner_stats.get('vf_loss', 0)
        
        # Get reward safely
        reward = policy_rewards.get(agent_id, 0)

        cum_reward += reward
        cum_loss += total_loss
        count += 1
        if show_each_agent:
            table_data.append([
                agent_id,
                f"{Fore.RED}{total_loss}{Style.RESET_ALL}",
                f"{Fore.YELLOW}{policy_loss}{Style.RESET_ALL}",
                f"{Fore.GREEN}{vf_loss}{Style.RESET_ALL}",
                f"{Fore.CYAN}{reward}{Style.RESET_ALL}"
            ])

    # Print the table if enabled
    if show_each_agent:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    print(Fore.MAGENTA + f"ITERATION {iteration_index} FINISHED, with Reward {cum_reward}, Loss {cum_loss}" + Style.RESET_ALL)
    print(Fore.BLACK + "______________________________________" + Style.RESET_ALL)

    return cum_loss/count, cum_reward/count # Return average loss and reward
