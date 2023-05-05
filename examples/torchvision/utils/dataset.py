import math


def get_num_iterations(dataset, batch_size, world_size):
    num_iterations = math.ceil(len(dataset) / batch_size / world_size)
    return num_iterations
