import numpy as np
from typing import Dict, Tuple, List
import sys

sys.path.append("..")  # Adjust path to import GridWorld
from Envs.GridWorld import GridWorld


class ValueIterationSolver:
    """
    Value Iteration Algorithm for solving MDPs

    Implements the Bellman equation:
    V(s) = max_a Σ P(s'|s,a) [R(s,a,s') + γV(s')]
    """

    def __init__(self, env: GridWorld, gamma: float = 0.9):
        self.env = env
        self.gamma = gamma
        self.values = {}
        self.policy = {}
        self.iteration_count = 0

        # Initialize value function to zero
        for state in env.states:
            self.values[state] = 0.0

        print(f"🧠 Value Iteration Solver Created!")
        print(f"Discount factor (γ): {gamma}")
        print(f"Convergence threshold: 1e-6")
        print()

    def compute_q_value(self, state: Tuple[int, int], action: int) -> float:
        """
        Compute Q(s,a) = Σ P(s'|s,a) [R(s,a,s') + γV(s')]
        """
        if self.env.is_terminal(state):
            return 0.0

        q_value = 0.0

        # Get all possible next states and their probabilities
        transitions = self.env.get_next_states(state, action)

        for next_state, prob in transitions:
            reward = self.env.get_reward(state, action, next_state)
            q_value += prob * (reward + self.gamma * self.values[next_state])

        return q_value

    def compute_state_value(self, state: Tuple[int, int]) -> float:
        """
        Compute V(s) = max_a Q(s,a)
        """
        if self.env.is_terminal(state):
            return 0.0

        valid_actions = self.env.get_valid_actions(state)
        if not valid_actions:
            return 0.0

        # Compute Q-value for each action and take maximum
        q_values = [self.compute_q_value(state, action) for action in valid_actions]
        return max(q_values)

    def value_iteration(
        self, theta: float = 1e-6, max_iterations: int = 1000
    ) -> Dict[Tuple[int, int], float]:
        """
        Main Value Iteration Algorithm

        Args:
            theta: Convergence threshold
            max_iterations: Maximum number of iterations

        Returns:
            Converged value function
        """
        print("🚀 Starting Value Iteration...")
        print("Iteration | Max Change | Converged?")
        print("-" * 35)

        for iteration in range(max_iterations):
            # Store old values for convergence check
            old_values = self.values.copy()

            # Update all state values using Bellman equation
            for state in self.env.states:
                self.values[state] = self.compute_state_value(state)

            # Check convergence
            max_change = max(
                abs(self.values[state] - old_values[state]) for state in self.env.states
            )

            converged = max_change < theta

            print(
                f"   {iteration:2d}     |   {max_change:.6f}  |   {'✅' if converged else '❌'}"
            )

            if converged:
                print(f"\n🎯 Converged after {iteration + 1} iterations!")
                break

        else:
            print(f"\n⚠️  Reached maximum iterations ({max_iterations})")

        self.iteration_count = iteration + 1
        return self.values

    def extract_policy(self) -> Dict[Tuple[int, int], int]:
        """
        Extract optimal policy from value function
        π*(s) = argmax_a Q(s,a)
        """
        print("\n🎯 Extracting Optimal Policy...")

        self.policy = {}

        for state in self.env.states:
            if self.env.is_terminal(state):
                continue

            valid_actions = self.env.get_valid_actions(state)
            if not valid_actions:
                continue

            # Find action with highest Q-value
            best_action = None
            best_q_value = float("-inf")

            for action in valid_actions:
                q_value = self.compute_q_value(state, action)
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action

            self.policy[state] = best_action

        return self.policy

    def print_results(self):
        """Print detailed results of value iteration"""
        print("\n📊 VALUE ITERATION RESULTS")
        print("=" * 50)

        # Print state values
        print("\n🎯 State Values V*(s):")
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                state = (r, c)
                if state == self.env.start_state:
                    marker = "S"
                elif state == self.env.goal_state:
                    marker = "G"
                elif state == self.env.pit_state:
                    marker = "#"
                else:
                    marker = " "

                value = self.values[state]
                print(f"  {marker}({r},{c}): {value:8.3f}")

        # Print policy
        print(f"\n🎯 Optimal Policy π*(s):")
        action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        action_arrows = ["↑", "↓", "←", "→"]

        for state, action in self.policy.items():
            r, c = state
            print(f"  ({r},{c}): {action_names[action]:>5} ({action_arrows[action]})")

        print(f"\n📈 Algorithm Performance:")
        print(f"  Iterations to converge: {self.iteration_count}")
        print(f"  Discount factor (γ): {self.gamma}")
        print(f"  Value at start state: {self.values[self.env.start_state]:.3f}")

    def simulate_episode(self, max_steps: int = 20) -> List[Tuple[int, int]]:
        """
        Simulate an episode using the learned policy
        """
        print("\n🎮 Simulating Episode with Optimal Policy...")

        current_state = self.env.start_state
        path = [current_state]

        for step in range(max_steps):
            if self.env.is_terminal(current_state):
                break

            # Get optimal action
            if current_state not in self.policy:
                print(f"  No policy for state {current_state}!")
                break

            action = self.policy[current_state]
            action_name = self.env.actions[action]

            # For simulation, use deterministic transitions (intended direction)
            next_state = self.env._move(current_state, action)

            print(f"  Step {step + 1}: {current_state} → {action_name} → {next_state}")

            current_state = next_state
            path.append(current_state)

        # Calculate total reward
        total_reward = 0
        for i in range(len(path) - 1):
            reward = self.env.get_reward(
                path[i], self.policy.get(path[i], 0), path[i + 1]
            )
            total_reward += reward

        if current_state == self.env.goal_state:
            print(f"  🎉 Reached Goal! Total reward: {total_reward}")
        elif current_state == self.env.pit_state:
            print(f"  💥 Fell in Pit! Total reward: {total_reward}")
        else:
            print(f"  ⏰ Max steps reached. Total reward: {total_reward}")

        return path


# Main execution and testing
if __name__ == "__main__":
    print("🔥 VALUE ITERATION ALGORITHM TEST")
    print("=" * 50)

    # Create environment
    env = GridWorld()

    # Create solver
    solver = ValueIterationSolver(env, gamma=0.9)

    # Run value iteration
    final_values = solver.value_iteration()

    # Extract optimal policy
    optimal_policy = solver.extract_policy()

    # Print detailed results
    solver.print_results()

    # Visualize results
    print("\n🎨 VISUALIZATIONS")
    print("=" * 30)

    print("📊 Grid with Optimal Values:")
    env.visualize(values=final_values)

    print("🎯 Grid with Optimal Policy:")
    env.visualize(policy=optimal_policy)

    print("🎯 Grid with Both Values and Policy:")
    env.visualize(values=final_values, policy=optimal_policy)

    # Simulate optimal episode
    solver.simulate_episode()

    print("\n🔥 VALUE ITERATION COMPLETE!")
    print(
        "🎉 We found the optimal policy! The agent now knows the best action in every state!"
    )

    # Test with different gamma values
    print("\n🧪 Testing Different Discount Factors:")
    for gamma in [0.5, 0.9, 0.99]:
        print(f"\n--- γ = {gamma} ---")
        test_solver = ValueIterationSolver(env, gamma=gamma)
        test_solver.value_iteration(theta=1e-6)
        start_value = test_solver.values[env.start_state]
        print(f"Start state value: {start_value:.3f}")
        print(f"Convergence iterations: {test_solver.iteration_count}")

    print("\n💪 Ready for Deep RL! This is the foundation everything builds on!")
