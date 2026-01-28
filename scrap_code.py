

def disparity_calc(data_a, data_b):
    num_true_0 = sum((1 for i in range(len(data_a)) if data_a[i] == True and data_b[i][0] == 0))
    num_true_1 = sum((1 for i in range(len(data_a)) if data_a[i] == True and data_b[i][0] == 1))

    probability_a = P(f(x) = 1 and group = 0)/(count(data_a == 0)/len(data_a)))
    probability_b = P(f(x) = 1 and group = 1)/P(group=1)

    disparity = abs(num_true_0 - num_true_1)