import pandas as pd
import numpy as np
from tqdm import trange


LEVEL_TO_PROBS = {
    4: [.55, .3, .15, 0, 0],
    5: [.45, .33, .2, .02, 0],
    6: [.25, .4, .3, .05, 0],
    7: [.19, .3, .35, .15, .01],
    8: [.15, .2, .35, .25, .05],
    9: [.1, .15, .3, .3, .15]
}

TIERS = [1, 2, 3, 4, 5]

CHAMPIONS_PER_TIER = {
    1: 13,
    2: 13,
    3: 13,
    4: 11,
    5: 10
}


class Simulator:
    def __init__(self, success_criteria, level=5, copies_already_held: dict={}, champions_missing:dict ={}) -> None:
        self.unweighted_probs = {
            1: [29]*13,
            2: [22]*13,
            3: [18]*13,
            4: [11]*12,
            5: [10]*7
        }
        self.success_criteria = success_criteria
        self.champions_missing = champions_missing
        for tier, champion in copies_already_held:
            self.unweighted_probs[tier][champion] -= copies_already_held[(tier, champion)]
        self.level = level

    def remove_champions(self):
        for tier in self.champions_missing:
            keys_of_tier = [key for key in self.success_criteria if ]
    def normalize_unweighted(self, probs):
        denom = sum(probs)
        return [prob/denom for prob in probs]

    def reset_probabilities(self):
        self.unweighted_probs = {
            1: [29]*13,
            2: [22]*13,
            3: [18]*13,
            4: [11]*12,
            5: [10]*7
        }

    def get_card(self):
        tier_probabilities = LEVEL_TO_PROBS[self.level]
        tier = np.random.choice(TIERS, p=tier_probabilities)
        unweighted_for_tier = self.unweighted_probs[tier]
        champion_probabilities = self.normalize_unweighted(unweighted_for_tier)
        return tier, np.random.choice(range(len(unweighted_for_tier)), p=champion_probabilities)


    def get_hand(self, success_dict):
        unwanted = []
        for _ in range(5):
            tier, champion = self.get_card()
            # print(tier, champion)
            self.unweighted_probs[tier][champion] -= 1
            if (tier, champion) not in success_dict:
                unwanted.append((tier, champion))
            else:
                success_dict[(tier, champion)] += 1

        # print(self.unweighted_probs)
        # print(success_dict)
        
        for tier, champion in unwanted: # put champions back into the shop if we didn't pick them up.
            self.unweighted_probs[tier][champion] += 1

        # print(self.unweighted_probs)


    def is_successful(self, success_dict, success_criteria):
        successes = zip(success_dict.values(), success_criteria.values())
        if all(current > expected for current, expected in successes):
            return True
        return False

    def simulate_once(self, success_criteria, num_gold): # dict of number of each champion desired.
        success_dict = {champion: 0 for champion in success_criteria.keys()}
        for _ in range(num_gold // 2):
            self.get_hand(success_dict)
            # print(success_dict)
            if self.is_successful(success_dict, success_criteria):
                self.reset_probabilities()
                return 1
        if self.is_successful(success_dict, success_criteria):
            self.reset_probabilities()
            return 1
        self.reset_probabilities()
        return 0


    def simulate(self, success_criteria, num_gold, n=10000):
        return sum([self.simulate_once(success_criteria, num_gold) for _ in range(n)])
        

if __name__ == '__main__':
    N = 10000
    probs = {}
    for level in [4, 5]:
        copies_already_held =  {(1, 0): 6}
        sim = Simulator(level=level, copies_already_held=copies_already_held)
        probs[level] = []
        success_criteria =  {(1, 0): 3}
        for i in trange(20, 50, 2):
            prob = sim.simulate(success_criteria, i, n=N) / N
            probs[level].append(prob)
            print(f'level: {level}, {i}: {prob}')

    pd.DataFrame(probs).to_csv('gold.csv', index=False)
        
