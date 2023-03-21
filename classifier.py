# Author: Samuel Hetteš, ID: 110968
# Subject: Artificial Intelligence
# Assignment: Classification using k-NN algorithm
# IDE: PyCharm 2021.2.2
# Programming language: Python 3.9
# Date: 29.11.2021

import math  # calculating euclid distance
import random  # generating random points
import matplotlib.pyplot as plt  # plotting results
import numpy as np  # finding k smallest values
from timeit import default_timer as timer  # measuring time
from collections import Counter  # finding most frequent value


# adds a point to the appropriate square
def add_to_squares(point, squares):
    x_index, y_index = int((4999 + point[0]) / 100), int((4999 + point[1]) / 100)  # calculate square indexes
    if squares[x_index][y_index][0] is None:  # if square is empty with None only, replace it
        squares[x_index][y_index][0] = point
    else:  # else append to existing values
        squares[x_index][y_index].append(point)


# returns euclid distance between points a and b
def euclid_dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# classifies point
def classify(x, y, k, points, squares_set):
    if squares_set:  # if set, search the map by squares
        x_index, y_index = int((4999 + x) / 100), int((4999 + y) / 100)  # calculate square indexes
        points_found = []  # list of all points found around our classified point
        squares_searched = []  # list of all squares that were already searched
        border = 1  # value to sub/add from/to original indexes
        while len(points_found) < k:  # while enough points are not found, search adjacent squares
            x_sequence = list(range(x_index - border, x_index + border + 1))  # create a sequence of x indexes
            y_sequence = list(range(y_index - border, y_index + border + 1))  # create a sequence of y indexes

            # cycle to search all adjacent squares - all combinations of x and y in sequences
            for x_temp in x_sequence:
                for y_temp in y_sequence:
                    # if square was already searched or indexes are out of range, continue
                    if (x_temp, y_temp) in squares_searched or not 0 <= x_temp < 100 or not 0 <= y_temp < 100:
                        continue
                    # if square is not empty, append all points in square to found points
                    if points[x_temp][y_temp][0] is not None:
                        for point in points[x_temp][y_temp]:
                            points_found.append(point)
                    squares_searched.append((x_temp, y_temp))  # add square to searched squares
            border += 1  # increase border
    else:  # else search all the points
        points_found = points

    point_count = len(points_found)  # number of points found
    if point_count == 1:  # if only one point was found, return its class
        return points_found[0][2]

    distances = np.array([None] * point_count)  # array of distances with points class
    for i in range(point_count):  # calculate distances for all points
        distances[i] = (euclid_dist((points_found[i][0], points_found[i][1]), (x, y)), points_found[i][2])

    if point_count != k:  # if number of points is not the same as value k
        distances = np.partition(distances, k)[:k]  # find and get k smallest values

    data = Counter([x[1] for x in distances])
    most_common = data.most_common()  # find most common values (classes)
    len_most_common = len(most_common)  # length of most common classes (number of classes)

    # if only one class is present or count of the most common class is unique return it
    if len_most_common == 1 or most_common[0][1] != most_common[1][1]:
        return most_common[0][0]

    # create an array for classes with same count
    if len_most_common > 2 and most_common[0][1] == most_common[2][1]:
        total_distances = [[0, most_common[0][0]], [0, most_common[1][0]], [0, most_common[2][0]]]
        len_most_common = 3
    else:
        total_distances = [[0, most_common[0][0]], [0, most_common[1][0]]]
        len_most_common = 2

    # calculate total distances for classes with same count
    for distance, category in distances:
        for i in range(len_most_common):
            if total_distances[i][1] == category:
                total_distances[i][0] += distance

    return min(total_distances)[1]  # return the class with smallest total distance


# plots and saves scatter plot figure of points
def plot_scatter(k, squares, fill_spaces):
    flattened_squares = []
    for x in range(100):  # flatten the array
        for y in range(100):
            if squares[x][y][0] is not None:
                for point in squares[x][y]:
                    flattened_squares.append(point)

    flattened_squares = sorted(flattened_squares, key=lambda z: [z[1], z[0]])  # sort points
    point_count, index = len(flattened_squares), 0

    if fill_spaces in {'y', 'Y'}:  # if set
        # classify and plot all the points that are missing
        for i in range(-5000, 5001):
            missing_points = []
            for j in range(-5000, 5001):
                if index < point_count and flattened_squares[index][0] == j and flattened_squares[index][1] == i:
                    index += 1
                else:
                    missing_points.append((i, j, classify(i, j, k, squares, 1)))
            plt.scatter([point[0] for point in missing_points], [point[1] for point in missing_points], 1,
                        [point[2] for point in missing_points])

    # plot points that were already generated as well and save figure
    plt.title(f'Classification for k = {k}')
    plt.scatter([point[0] for point in flattened_squares], [point[1] for point in flattened_squares], 1,
                [point[2] for point in flattened_squares])
    plt.savefig(f'k{k}.png')
    plt.close()


# main function
def main():
    print('******************************************')
    print('>>>             CLASSIFIER             <<<')
    print('******************************************\n')

    # input
    iterations = int(input("Enter the number of points to generate: "))
    fill_spaces = input("Fill white spaces? [y/n]: ")

    # initial sample
    sample = [
        (-4500, -4400, 'r'), (-4100, -3000, 'r'), (-1800, -2400, 'r'), (-2500, -3400, 'r'), (-2000, -1400, 'r'),
        (4500, -4400, 'g'), (4100, -3000, 'g'), (1800, -2400, 'g'), (2500, -3400, 'g'), (2000, -1400, 'g'),
        (-4500, 4400, 'b'), (-4100, 3000, 'b'), (-1800, 2400, 'b'), (-2500, 3400, 'b'), (-2000, 1400, 'b'),
        (4500, 4400, 'm'), (4100, 3000, 'm'), (1800, 2400, 'm'), (2500, 3400, 'm'), (2000, 1400, 'm')
    ]

    # 3D lists with values divided into squares of size 100*100 -> 10000 squares and 1D lists of classified points
    k1_squares, k1_points = [[[None] for y in range(100)] for x in range(100)], [row[:] for row in sample]
    k3_squares, k3_points = [[[None] for y in range(100)] for x in range(100)], [row[:] for row in sample]
    k7_squares, k7_points = [[[None] for y in range(100)] for x in range(100)], [row[:] for row in sample]
    k15_squares, k15_points = [[[None] for y in range(100)] for x in range(100)], [row[:] for row in sample]
    k1_correct, k3_correct, k7_correct, k15_correct = 0, 0, 0, 0  # counts of correctly classified points

    for i in range(20):  # add all initial points to appropriate squares
        for squares in k1_squares, k3_squares, k7_squares, k15_squares:
            add_to_squares(sample[i], squares)

    start_time = timer()  # starting time
    for i in range(iterations):  # generating points
        remainder = i % 4  # points are generated in order: Red, Green, Blue, Purple
        if remainder == 0:  # Red: x < 500, y < 500
            category = 'r'
            new = (random.randint(-5000, 499), random.randint(-5000, 499), category)
        elif remainder == 1:  # Green: x > -500, y < 500
            category = 'g'
            new = (random.randint(-499, 5000), random.randint(-5000, 499), category)
        elif remainder == 2:  # Blue: x < 500, y > -500
            category = 'b'
            new = (random.randint(-5000, 499), random.randint(-499, 5000), category)
        else:  # Purple: x > -500, y > -500
            category = 'm'
            new = (random.randint(-499, 5000), random.randint(-499, 5000), category)

        random_num = random.randint(1, 100)  # random number from range (1-100)
        if random_num == 1:  # 1% chance of generating a new point throughout the whole space
            new = (random.randint(-5000, 5000), random.randint(-5000, 5000), category)

        # classify points for each k value
        if i < 1000:  # for the first 1000 points search the whole map
            k1_class = classify(new[0], new[1], 1, k1_points, 0)
            k3_class = classify(new[0], new[1], 3, k3_points, 0)
            k7_class = classify(new[0], new[1], 7, k7_points, 0)
            k15_class = classify(new[0], new[1], 15, k15_points, 0)
            if i == 999:  # clear the arrays after the 1000th point, we are not going to use them anymore
                del k1_points
                del k3_points
                del k7_points
                del k15_points
        else:  # when 1000 points are generated, start searching by squares
            k1_class = classify(new[0], new[1], 1, k1_squares, 1)
            k3_class = classify(new[0], new[1], 3, k3_squares, 1)
            k7_class = classify(new[0], new[1], 7, k7_squares, 1)
            k15_class = classify(new[0], new[1], 15, k15_squares, 1)

        # check if points were classified correctly
        if k1_class == category:  # k = 1
            k1_correct += 1
        if k3_class == category:  # k = 3
            k3_correct += 1
        if k7_class == category:  # k = 7
            k7_correct += 1
        if k15_class == category:  # k = 15
            k15_correct += 1

        if new not in sample:  # if generated point is not in sample
            sample.append(new)  # add generated point to sample
            if i < 999:  # append only the first 999 points
                k1_points.append((new[0], new[1], k1_class))
                k3_points.append((new[0], new[1], k3_class))
                k7_points.append((new[0], new[1], k7_class))
                k15_points.append((new[0], new[1], k15_class))
            for (cls, squares) in (k1_class, k1_squares), (k3_class, k3_squares), (k7_class, k7_squares),\
                                  (k15_class, k15_squares):  # add points to squares
                add_to_squares((new[0], new[1], cls), squares)
    end_time = timer()  # ending time
    del sample

    # printing results
    print(f'\nPoints generated: {iterations}')
    print(f'Total time taken: {round(end_time - start_time, 2)} sec')
    for (correct, k) in (k1_correct, 1), (k3_correct, 3), (k7_correct, 7), (k15_correct, 15):
        print('------------------------------------------')
        print(f'>>> Results for k = {k}')
        print(f'Classified correctly: {correct}')
        print(f'Classified incorrectly: {iterations - correct}')
        print(f'Success rate: {round(correct / (iterations / 100), 2)}%')

    for (k, squares) in (1, k1_squares), (3, k3_squares), (7, k7_squares), (15, k15_squares):  # plotting classifiers
        plot_scatter(k, squares, fill_spaces)


if __name__ == '__main__':  # run main
    main()
