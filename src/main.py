from environment import Environment, HistoricData
from agent import Agent, PPOMemory

if __name__ == "__main__":
    history = HistoricData("datos/clean/merged_data.csv")
    env = Environment(history)
    agent = Agent(n_actions=1, input_dims=(73,))
    n_episodes = 2
    for i_episode in range(n_episodes):
        observation = env.reset()  # Reset the environment
        done = False

        while not done:
            action = agent.choose_action(observation)  # Agent chooses an action

            # Perform the action in the environment
            new_observation, reward, done = env.step(action)

            # Store the experience in memory
            agent.memory.store_memory_from_state(observation, action, reward, done)

            observation = new_observation  # Update the state

            # If enough experiences have been stored, perform a learning step
            if agent.memory.mem_cntr % 256 == 0:
                agent.learn()

        print("Episode finished")
        # Print progress every 10 episodes
        if i_episode % 10 == 0:
            print(f"Episode {i_episode}/{n_episodes}")
    print("Training finished")
    agent.save_models()
    for i in range(10):
        observation = env.reset()
        done = False
        while not done:
            action = agent.choose_action(observation)
            print(f"Value for delta(B): {action}")
            new_observation, reward, done = env.step(action)
            observation = new_observation
        print("Episode finished")
