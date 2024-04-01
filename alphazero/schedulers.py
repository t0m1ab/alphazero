from alphazero.base import TemperatureScheduler


class ConstantTemperatureScheduler(TemperatureScheduler):
    """
    Constant temperature scheduler used to control the exploration of the MCTS.
    * temp = temp_max_step for all steps
    """

    def __init__(self, temp_max_step: int, temp_min_step: int, max_steps: int) -> None:
        super().__init__(temp_max_step, temp_min_step, max_steps)
        if self.temp_min_step != self.temp_max_step:
            raise ValueError("temp_min_step should be equal to temp_max_step for constant scheduler.")
    
    def compute_temperature(self, step: int) -> float:
        """ Compute the temperature at the given step. """
        return 0


class LinearTemperatureScheduler(TemperatureScheduler):
    """
    Linear temperature scheduler used to control the exploration of the MCTS.
    * temp = 1 for steps in [0, temp_max_step]
    * temp = linear interpolation between 1 and 0 for steps in [temp_max_step+1, temp_min_step-1]
    * temp = 0 for steps in [temp_min_step, max_step]
    """

    def __init__(self, temp_max_step: int, temp_min_step: int, max_steps: int) -> None:
        super().__init__(temp_max_step, temp_min_step, max_steps)
        if self.temp_min_step < self.temp_max_step:
            raise ValueError("temp_min_step should be greater than temp_max_step for linear scheduler.")
    
    def compute_temperature(self, step: int) -> float:
        """ Compute the temperature at the given step. """
        if step <= self.temp_max_step:
            return 1
        elif step >= self.temp_min_step:
            return 0
        else:
            return 1 - (step - self.temp_max_step) / (self.temp_min_step - self.temp_max_step)


def main():
    
    _ = ConstantTemperatureScheduler(10, 10, 100)
    print("ConstantTemperatureScheduler created successfully.")

    _ = LinearTemperatureScheduler(10, 20, 100)
    print("LinearTemperatureScheduler created successfully.")


if __name__ == "__main__":
    main()