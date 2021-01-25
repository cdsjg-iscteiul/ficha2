import random as rand
from functools import partial
import time
import matplotlib.pyplot as plt

n_figure = 1

seed = time.time()
print(seed)
rand.seed(1611491668.914867)


# Rule: number of 10 in the pattern
# higher number of points in:
# 8: 4
# 16: 8
# 32: 16
# 64: 32
# 128: 64
# 256: 128

# PERGUNTA 1
def generate(x):
    string = ""
    for i in range(x):
        tmp = str(rand.randint(0, 1))
        string += tmp

    return string


# PERGUNTA 2 - INICIO
def guess(pattern, x):
    pattern_guess = ""
    number_tries = 0

    while pattern_guess != pattern:
        pattern_guess = ""
        for i in range(x):
            tmp = str(rand.randint(0, 1))
            pattern_guess += tmp
        number_tries += 1

    return number_tries


def new_start():
    bits = [8, 16]

    number_tent_8 = []
    number_tent_16 = []

    times_8 = []
    times_16 = []

    for t in range(30):

        for x in bits:
            start_time = time.time()
            size = generate(x)
            number_tries = guess(size, x)
            if x == 8:
                number_tent_8.append(number_tries)
                time_8 = (time.time() - start_time)
                times_8.append(time_8)

            if x == 16:
                number_tent_16.append(number_tries)
                time_16 = (time.time() - start_time)
                times_16.append(time_16)

    make_graph([number_tent_8, number_tent_16], [times_8, times_16], bits)


def make_graph(plot1, plot2, bits):
    global n_figure
    fig, ax1 = plt.subplots(1, 1)
    ax1.boxplot(plot1, labels=bits, vert=True)
    ax1.legend(['Figure ' + str(n_figure)], handlelength=0)
    n_figure += 1
    ax1.set_title('Bit Numbers vs Number of attempts')

    fig, ax2 = plt.subplots(1, 1)
    ax2.boxplot(plot2, labels=bits, vert=True)
    ax2.legend(['Figure ' + str(n_figure)], handlelength=0)
    n_figure += 1
    ax2.set_title('Bit Numbers vs Time')

    plt.show()


# FIM PERGUNTA 2

# INICIO PERGUNTA 3
def evaluation(testing, target):
    target = list(target)
    testing = list(testing)
    correct = 0

    for x in range(len(target)):
        if target[x] == testing[x]:
            correct += 1

    return correct / len(target)


def remove_wrong(testing, target):
    target = list(target)
    testing = list(testing)

    for x in range(len(target)):
        if testing[x] != target[x]:
            testing[x] = None

    return testing


def new_guess():
    bits = [8, 16, 32, 64, 128, 256]

    runtimes = [[] for _ in range(len(bits))]
    attempts = [[] for _ in range(len(bits))]

    for x in range(30):

        for y in bits:
            start_time = time.time()
            target = generate(y)
            testing = generate(y)
            numb_tent_current = 0
            while evaluation(target, testing) < 1:
                numb_tent_current += 1
                removed_wrong = remove_wrong(testing, target)
                for t in range(len(removed_wrong)):
                    if removed_wrong[t] is None:
                        removed_wrong[t] = str(rand.randint(0, 1))

                testing = "".join(removed_wrong)

            attempts[bits.index(y)].append(numb_tent_current)
            runtimes[bits.index(y)].append(time.time() - start_time)

    make_graph(attempts, runtimes, bits)


# FIM PERGUNTA 3

def rule_evaluation(testing):
    rule_test = list(testing)
    x = 0
    score = 0

    while x < len(rule_test):
        if rule_test[x] == "1" and rule_test[x + 1] == "0":
            score += 1
        x += 1


# INICIO PERGUNTA 4(2)
def mutation(mutate):
    for_mutation = list(mutate)
    random_spot = rand.randint(0, len(for_mutation) - 1)

    if for_mutation[random_spot] == '1':
        for_mutation[random_spot] = "0"

    else:
        for_mutation[random_spot] = "1"

    return "".join(for_mutation)


def final_mutation():
    bits = [8, 16, 32, 64, 128, 256]

    runtimes = [[] for _ in range(len(bits))]
    attempts = [[] for _ in range(len(bits))]

    for t in range(30):
        for x in bits:
            start_time = time.time()
            target = generate(x)
            testing = generate(x)
            num_tent = 0

            for y in range(1000):
                num_tent += 1
                if evaluation(target, testing) == 1:
                    break

                mid_test = mutation(testing)

                if evaluation(target, mid_test) > evaluation(target, testing):
                    testing = mid_test

            attempts[bits.index(x)].append(num_tent)
            runtimes[bits.index(x)].append(time.time() - start_time)

    make_graph(attempts, runtimes, bits)


# FIM PERGUNTA 4 2

# PERGUNTA 5

def sortGeneration(generation, target):
    return sorted(generation, key=partial(evaluation, target=target))


def mutated_pop():
    bits = [8, 16, 32, 64, 128, 256]
    limit_stagnation = 10

    runtimes = [[] for _ in range(len(bits))]
    attempts = [[] for _ in range(len(bits))]

    for t in range(30):

        for x in bits:
            start_time = time.time()
            target = generate(x)
            population = []
            best_pop = []

            best_ev = 0
            times_equal = 0
            nr_generations = 0

            for _ in range(100):
                population.append(generate(x))

            population = sortGeneration(population, target)

            while times_equal < limit_stagnation:
                nr_generations += 1
                best_pop = population[:30]

                if evaluation(best_pop[0], target) == best_ev:
                    times_equal += 1
                else:
                    times_equal = 0

                best_ev = evaluation(best_pop[0], target)

                while len(best_pop) < 100:
                    spot = rand.choice(best_pop)
                    mutated = mutation(spot)
                    best_pop.append(mutated)

                population = sortGeneration(best_pop, target)

            attempts[bits.index(x)].append(nr_generations)
            runtimes[bits.index(x)].append(time.time() - start_time)

    make_graph(attempts, runtimes, bits)


# FIM PERGUNTA 5

# inicio pergunta 6

def new_crossover(father, mother):
    split = rand.randint(0, len(father))

    part_father = father[:split]
    part_mother = mother[split:]

    return part_father + part_mother


def crossoverITALL():
    bits = [8, 16, 32, 64, 128, 256]
    limit_stagnation = 10

    runtimes = [[] for _ in range(len(bits))]
    attempts = [[] for _ in range(len(bits))]

    for t in range(30):

        for x in bits:
            start_time = time.time()
            target = generate(x)
            population = []
            best_pop = []

            best_ev = 0
            times_equal = 0
            nr_generations = 0

            for _ in range(100):
                population.append(generate(x))

            population = sortGeneration(population, target)

            while times_equal < limit_stagnation:
                nr_generations += 1
                best_pop = population[:30]

                if evaluation(best_pop[0], target) == best_ev:
                    times_equal += 1
                else:
                    times_equal = 0

                best_ev = evaluation(best_pop[0], target)

                while len(best_pop) < 100:
                    spot_dad = rand.choice(best_pop)
                    spot_mom = rand.choice(best_pop)
                    son = new_crossover(spot_dad, spot_mom)
                    best_pop.append(son)

                population = sortGeneration(best_pop, target)

            attempts[bits.index(x)].append(nr_generations)
            runtimes[bits.index(x)].append(time.time() - start_time)

    make_graph(attempts, runtimes, bits)


# fim pergunta 6

new_start()
new_guess()
final_mutation()
mutated_pop()
crossoverITALL()
