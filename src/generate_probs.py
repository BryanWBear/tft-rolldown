import pandas as pd
import numpy as np
from tqdm import trange, tqdm
import seaborn as sns
import matplotlib.pyplot as plt

LEVEL_TO_PROBS = {
    4: [.55, .3, .15, 0, 0],
    5: [.45, .33, .2, .02, 0],
    6: [.25, .4, .3, .05, 0],
    7: [.19, .3, .35, .15, .01],
    8: [.16, .2, .35, .25, .04],
    9: [.09, .15, .3, .3, .16]
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
    def __init__(self, success_criteria, level=5, copies_already_held: dict={}, champions_missing:dict ={}, semantics='and', debug=False) -> None: # TODO: can consolidate copies held / champs missing.
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
        self.remove_champions()
        print(self.unweighted_probs)
        self.level = level
        self.semantics = semantics
        self.debug = debug
        assert self.semantics in ['and', 'or']

    def remove_champions(self):
        for tier in self.champions_missing:
            max_champion_idx = max([champion for t, champion in self.success_criteria if t == tier]) + 1
            removed_count = 0
            while removed_count < self.champions_missing[tier]:
                if self.unweighted_probs[tier][max_champion_idx] == 0:
                    max_champion_idx += 1
                self.unweighted_probs[tier][max_champion_idx] -= 1
                removed_count += 1

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
            if self.debug: # too lazy to use a logging library
                print(f'tier, champion drawn: {tier}, {champion}')
            # print(tier, champion)
            self.unweighted_probs[tier][champion] -= 1
            if self.debug:
                print(f'unweighted probs after draw: {self.unweighted_probs}')
            if (tier, champion) not in success_dict:
                unwanted.append((tier, champion))
            else:
                success_dict[(tier, champion)] += 1
                if self.debug:
                    print(f'success dict after draw: {success_dict}')

        # print(self.unweighted_probs)
        # print(success_dict)
        
        for tier, champion in unwanted: # put champions back into the shop if we didn't pick them up.
            self.unweighted_probs[tier][champion] += 1
        
        if self.debug:
            print(f'unweighted probs after putting back: {self.unweighted_probs}')

        # print(self.unweighted_probs)


    def is_successful(self, success_dict, success_criteria):
        successes = list(zip(success_dict.values(), success_criteria.values()))
        if self.semantics == 'and':
            if all(current >= expected for current, expected in successes):
                if self.debug:
                    print(f'success dict: {success_dict}')
                    print(f'success_criteria: {success_criteria}')
                    print(f'success dict values zipped: {successes}')
                return True
            return False
        elif self.semantics == 'or':
            if any(current >= expected for current, expected in successes):
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

    def simulate_dist_once(self, success_criteria): # dict of number of each champion desired.
        success_dict = {champion: 0 for champion in success_criteria.keys()}
        hands = 1
        while not self.is_successful(success_dict, success_criteria):
            self.get_hand(success_dict)
            # print(success_dict)
            hands += 1
        self.reset_probabilities()
        return hands


    def simulate_dist(self, success_criteria, n=10000):
        tries = []
        for _ in trange(n):
            tries.append(self.simulate_dist_once(success_criteria))
        return tries
        

if __name__ == '__main__':
    N = 1
    probs = {}
    for level in [7]:
        success_criteria =  {(3, 0): 1} #, (4, 1): 3, (4, 2): 3, (4, 3): 3}
        copies_already_held =  {(3, 0): 2}
        champions_missing = {} # {1: 40, 2: 20, 3: 10}
        sim = Simulator(success_criteria, level=level, copies_already_held=copies_already_held, champions_missing=champions_missing, semantics='and', debug=True)
        # probs[level] = []
        # for i in trange(20, 70, 2):
        #     prob = sim.simulate(success_criteria, i, n=N) / N
        #     probs[level].append(prob)
        #     print(f'level: {level}, {i}: {prob}')
        res = sim.simulate_dist(success_criteria, n=N)
        sns.displot(res, kde=True)
        plt.savefig('dist.png')
    # pd.DataFrame(probs).to_csv('4_carries_distribution_7_need_1.csv', index=False)
        
