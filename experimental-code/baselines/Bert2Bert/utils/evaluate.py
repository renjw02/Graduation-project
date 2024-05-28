
def evaluate_logic_form(ground_truth, prediction):
    if len(ground_truth) != len(prediction):
        raise 0.0
    
    correct = 0
    for gt, pred in zip(ground_truth, prediction):
        if gt == pred:
            correct += 1
    
    return correct / len(ground_truth)