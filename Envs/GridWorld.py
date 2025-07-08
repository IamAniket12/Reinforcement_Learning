import numpy as np
import random
from typing import List, Tuple, Dict


class GridWorld:
    """
    3x3 Grid World Environment for RL
    Layout:
    [S] [ ] [G]
    [ ] [#] [ ]
    [ ] [ ] [ ]

    S = Start (0,0), G = Goal (0,2), # = Pit (1,1)
    """

    def __init__(self):
        self.rows = 3
        self.cols = 3
        self.start_state = (0, 0)
        self.goal_state = (0, 2)
        self.pit_state = (1, 1)

        # Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.action_effects = {
            0: (-1, 0),  # UP
            1: (1, 0),  # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1),  # RIGHT
        }

        # All possible states
        self.states = [(r, c) for r in range(self.rows) for c in range(self.cols)]

        print("🎮 Grid World Environment Created!")
        print("Layout:")
        print("[S] [ ] [G]")
        print("[ ] [#] [ ]")
        print("[ ] [ ] [ ]")
        print("S=Start, G=Goal(+10), #=Pit(-10), Step=-1")
        print()

    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        """Check if state is within grid bounds"""
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """Check if state is terminal (goal or pit)"""
        return state == self.goal_state or state == self.pit_state

    def get_reward(
        self, state: Tuple[int, int], action: int, next_state: Tuple[int, int]
    ) -> float:
        """Get reward for transition"""
        if next_state == self.goal_state:
            return 10.0
        elif next_state == self.pit_state:
            return -10.0
        else:
            return -1.0  # Step cost

    def get_next_states(
        self, state: Tuple[int, int], action: int
    ) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get all possible next states with probabilities
        Returns: [(next_state, probability), ...]

        Stochastic transitions:
        - 80% intended direction
        - 10% perpendicular left
        - 10% perpendicular right
        """
        if self.is_terminal(state):
            return [(state, 1.0)]  # Terminal states stay put

        # Define perpendicular actions
        perp_actions = {
            0: [2, 3],  # UP -> LEFT, RIGHT
            1: [2, 3],  # DOWN -> LEFT, RIGHT
            2: [0, 1],  # LEFT -> UP, DOWN
            3: [0, 1],  # RIGHT -> UP, DOWN
        }

        transitions = []

        # Intended direction (80%)
        intended_next = self._move(state, action)
        transitions.append((intended_next, 0.8))

        # Perpendicular directions (10% each)
        for perp_action in perp_actions[action]:
            perp_next = self._move(state, perp_action)
            transitions.append((perp_next, 0.1))

        return transitions

    def _move(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Attempt to move in given direction"""
        r, c = state
        dr, dc = self.action_effects[action]
        next_state = (r + dr, c + dc)

        # If next state is invalid, stay in current state
        if not self.is_valid_state(next_state):
            return state

        return next_state

    def get_valid_actions(self, state: Tuple[int, int]) -> List[int]:
        """Get valid actions from current state"""
        if self.is_terminal(state):
            return []
        return list(range(len(self.actions)))

    def visualize(
        self,
        values: Dict[Tuple[int, int], float] = None,
        policy: Dict[Tuple[int, int], int] = None,
    ):
        """Visualize the grid world with optional values and policy"""
        print("\n🎮 Grid World Visualization")
        print("=" * 30)

        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                state = (r, c)

                # Special states
                if state == self.start_state:
                    cell = " S "
                elif state == self.goal_state:
                    cell = " G "
                elif state == self.pit_state:
                    cell = " # "
                else:
                    cell = "   "

                # Add values if provided
                if values and state in values:
                    cell = f"{values[state]:5.1f}"

                # Add policy arrows if provided
                if policy and state in policy:
                    arrows = ["↑", "↓", "←", "→"]
                    arrow = arrows[policy[state]]
                    cell = f" {arrow} "

                row_str += f"[{cell}]"

            print(row_str)

        print("=" * 30)

        if values:
            print(
                f"Value range: {min(values.values()):.2f} to {max(values.values()):.2f}"
            )
        print()


# Test the Grid World Environment
if __name__ == "__main__":
    # Create environment
    env = GridWorld()

    # Test basic functionality
    print("🔥 Testing Grid World Environment:")
    print()

    # Test state validation
    print("✅ Valid states:", env.is_valid_state((0, 0)), env.is_valid_state((2, 2)))
    print("❌ Invalid states:", env.is_valid_state((-1, 0)), env.is_valid_state((3, 3)))

    # Test terminal states
    print("🏁 Terminal states:", env.is_terminal((0, 2)), env.is_terminal((1, 1)))
    print("🏃 Non-terminal:", env.is_terminal((1, 0)))

    # Test transitions from start state
    print("\n🎯 Transitions from Start (0,0) with action RIGHT:")
    transitions = env.get_next_states((0, 0), 3)  # RIGHT
    for next_state, prob in transitions:
        reward = env.get_reward((0, 0), 3, next_state)
        print(f"  → {next_state} (prob: {prob:.1f}, reward: {reward:+.1f})")

    # Test transitions from middle state
    print("\n🎯 Transitions from Middle (1,0) with action UP:")
    transitions = env.get_next_states((1, 0), 0)  # UP
    for next_state, prob in transitions:
        reward = env.get_reward((1, 0), 0, next_state)
        print(f"  → {next_state} (prob: {prob:.1f}, reward: {reward:+.1f})")

    # Visualize empty grid
    print("\n🎮 Basic Grid Layout:")
    env.visualize()

    # Test with some sample values
    sample_values = {
        (0, 0): 8.2,
        (0, 1): 9.1,
        (0, 2): 0.0,
        (1, 0): 7.3,
        (1, 1): 0.0,
        (1, 2): 8.1,
        (2, 0): 6.4,
        (2, 1): 7.3,
        (2, 2): 8.2,
    }

    print("📊 Grid with Sample Values:")
    env.visualize(values=sample_values)

    # Test with sample policy
    sample_policy = {
        (0, 0): 3,
        (0, 1): 3,  # RIGHT
        (1, 0): 0,
        (1, 2): 0,  # UP
        (2, 0): 0,
        (2, 1): 0,
        (2, 2): 0,  # UP
    }

    print("🎯 Grid with Sample Policy:")
    env.visualize(policy=sample_policy)

    print("🔥 Grid World Environment Test Complete!")
    print("Ready for Value Iteration! 💪")
