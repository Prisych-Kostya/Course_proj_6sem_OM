class Function:

    def __init__(self, exp, starting_point=[0, 0], point_min=[0], min_value=0):
        """
        exp: expression that defines the function
        starting_point: starting point for the function
        point_min: point where the function reaches minimum value
        min_value: minimum value of the function
        """
        self.iterations = 0
        self.expression = exp
        self.starting_point = starting_point
        self.point_min = point_min
        self.min_value = min_value

    def reset_iterations(self):
        self.iterations = 0

    def calc(self, *args):
        """
        Calculates the value of the function for the given point.
        Function also track the number of calls.
        args: point
        returns: value of the function at the provided point
        """
        self.iterations += 1
        return self.expression(*args)