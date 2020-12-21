import random as rand
from functools import partial
import time
import matplotlib.pyplot as plt


# Rule: number of 10 in the pattern
# higher number of points in:
# 8: 4
# 16: 8
# 32: 16
# 64: 32
# 128: 64
# 256: 128

def generate(x):
    string = ""
    for i in range(x):
        tmp = str(rand.randint(0, 1))
        string += tmp

    return string


def generate_random(x):
    string = ""
    for i in range(x):
        tmp = str(rand.randint(0, 1))
        string += tmp

    return string


def guess(pattern, x):
    pattern_guess = ""
    number_tries = 0

    while pattern_guess != pattern:
        pattern_guess = ""
        for i in range(x):
            tmp = str(rand.randint(0, 1))
            pattern_guess += tmp

        (pattern, pattern_guess)
        evaluation(pattern, pattern_guess)
        number_tries += 1

    return number_tries


def start():
    number_8 = 0
    number_16 = 0
    number_32 = 0
    number_64 = 0
    number_128 = 0
    number_256 = 0
    all_tents = []

    number_tent_8 = []
    for x in range(30):
        # 8 bits
        size8 = generate(8)
        number_tries8 = guess(size8, 8)
        number_tent_8.append(number_tries8)
        number_8 = (number_8 + number_tries8) / 30
    all_tents.append(number_tent_8)
    print("DONE 8")
    print("TENTATIVAS MÉDIAS PARA 8 BITS:  " + str(number_8))
    make_graph(number_tent_8)
    number_tent_16 = []
    for x in range(30):
        # 16 bits
        size16 = generate(16)
        number_tries16 = guess(size16, 16)
        number_tent_16.append(number_tries16)
        number_16 = (number_16 + number_tries16) / 30
    all_tents.append(number_tent_16)
    print("DONE 16")
    print("TENTATIVAS MÉDIAS PARA 16 BITS:  " + str(number_16))

    make_graph(number_tent_16)


def make_graph(plot):
    fig = plt.figure(1, figsize=(9, 6))

    ax = fig.add_subplot(111)

    bp = ax.boxplot(plot)

    plt.show()


def evaluation(testing, target):
    splitted = list(target)
    testing = list(testing)
    correct = 0
    x = 0

    while x < len(splitted):
        if splitted[x] == testing[x]:
            correct += 1
        x += 1

    proximity = len(splitted) - correct
    return proximity


def rule_evaluation(testing):
    rule_test = list(testing)
    x = 0
    score = 0

    while x < len(rule_test):
        if rule_test[x] == "1" and rule_test[x + 1] == "0":
            score += 1
        x += 1


def mutation(target, mutate):
    for_mutation = list(mutate)
    the_best = list(target)
    best_score = 0
    best = []
    x = 0

    while x < 1000:
        random_spot = rand.randint(0, len(for_mutation) - 1)
        for_mutation[random_spot] = rand.randint(0, 1)
        check = evaluation("".join(list(map(str, for_mutation))), "".join(list(map(str, the_best))))
        if check > best_score:
            best_score = check
            best = for_mutation

        if check == best_score:
            print(check)
            print(best_score)
            print("".join(list(map(str, for_mutation))))
            return "BEST FOUND!!"

        x += 1

    return "".join(list(map(str, for_mutation)))


def new_mutation(target, mutate):
    for_mutation = list(mutate)
    the_best = list(target)
    best_score = 0
    best = []
    x = 0

    random = rand.randint(0, len(for_mutation) - 1)
    for_mutation[random] = rand.randint(0, 1)
    check = evaluation("".join(list(map(str, for_mutation))), "".join(list(map(str, the_best))))
    if check > best_score:
        best_score = check
    if check == best_score:
        return "BEST FOUND!!"

    x += 1

    return "".join(list(map(str, for_mutation)))


def population(target, size):
    populate = []
    best_list = []
    next_gen = []
    x = 0
    y = 0

    while x < 100:
        populate.append(generate_random(size))
        x += 1

    best_list = sortGeneration(populate, target)

    while y < 30:
        dummy = best_list.pop(y)
        next_gen.append(dummy)
        y += 1

    return next_gen


def population_with_mutation(mutate, target):
    mutated_population = []

    for x in mutate:
        if len(mutate) == 100:
            break
        else:
            in_mutation = rand.choice(mutate)
            mutated = mutation(target, in_mutation)
            if mutated == "BEST FOUND!!":
                return "STOP"
            else:
                mutated_population.append(mutated)

    return mutated_population


def crossover(generation):
    next_gen = []
    counter = 0

    while counter < len(generation) - 1:
        pick_father = generation[counter]
        pick_mother = generation[counter + 1]

        breakdown_mom = list(pick_mother)
        breakdown_dad = list(pick_father)

        son_1 = []
        son_2 = []

        t = 0

        while t < len(breakdown_dad):
            if t < (len(breakdown_dad) / 2):
                son_1.append(breakdown_dad[t])
                son_2.append(breakdown_mom[t])

            if t >= (len(breakdown_dad) / 2):
                son_2.append(breakdown_dad[t])
                son_1.append(breakdown_mom[t])

            t += 1

        next_gen.append("".join(list(map(str, son_1))))
        next_gen.append("".join(list(map(str, son_2))))

        counter += 1

    return next_gen


def sortGeneration(generation, target):
    return sorted(generation, key=partial(evaluation, target=target))


def checkIFFOUND(target, generation):
    for x in generation:
        if x == target:
            print(x)
            return "STOP"

    return 0


def generation_mutation():
    time_2 = 0
    time_1 = time.process_time()
    target = generate(64)
    print(target)
    next_gen = population(target, 64)
    stop = ""

    num_generations = 0

    while stop != "STOP":
        num_generations += 1
        mutated = population_with_mutation(next_gen, target)
        if mutated == "STOP":
            stop = "STOP"
            print("stop")
            time_2 = time.process_time()
        else:
            next_gen = population_with_mutation(mutated, target)

    print(num_generations)
    print("TIME:   " + str(time_2 - time_1))

def crossoverITALL():
    target = generate(16)
    print(target)
    next_gen = population(target, 16)
    stop = ""

    num_generations12 = 0

    while stop != "STOP":
        num_generations12 += 1
        sorted = sortGeneration(next_gen, target)
        print("MELHOR DA GERAÇÃO " + str(num_generations12) + ":  " + sorted[0] + "\n")
        print("SEGUNDO MELHOR DA GERAÇÃO " + str(num_generations12) + ":  " + sorted[1] + "\n")
        test = checkIFFOUND(target, sorted)
        crossover12 = crossover(sorted)
        if test == "STOP":
            stop = "STOP"
            print("stop")
        else:
            next_gen = crossover(crossover12)

    print(num_generations12)


crossoverITALL()
