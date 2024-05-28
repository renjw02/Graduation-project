import re
import sqlite3

from Evaluator import RestaurantsEvaluator, AdvisingEvaluator


def read_data(file_path='error_case.txt'):
    ground_truth = []
    predicted = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        for i in range(0, len(lines)-1, 3):  # 以步长3来读取每一组数据
            gt_line = lines[i].strip().replace("ground_truth: ", "")
            pd_line = lines[i+1].strip().replace("predicted: ", "")
            ground_truth.append(gt_line)
            predicted.append(pd_line)

            


    return ground_truth, predicted


if __name__ == '__main__':
    print("Read data...")
    file = './output/advising/results_25143027'
    data = read_data(file + '.txt')
    # restaurants_evaluator = RestaurantsEvaluator('./data/restaurants-db.added-in-2020.sqlite')
    advising_evaluator = AdvisingEvaluator('./data/advising-db.added-in-2020.sqlite')

    print("Start evaluating...")
    advising_evaluator.evaluate_execution(data[0], data[1], file)
    # restaurants_evaluator.evaluate_execution(data[0], data[1])

