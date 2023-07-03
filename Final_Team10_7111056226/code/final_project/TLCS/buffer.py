import random

class Buffer:
    # 緩存大小
    def __init__(self, size_max, size_min):
        self.experience = []
        self.size_max = size_max
        self.size_min = size_min

    # 增加sample進入buffer
    def add_sample(self, sample):
        self.experience.append(sample)
        if self.size_now() > self.size_max:
            self.experience.pop(0)

    def get_samples(self, n):
        if self.size_now() < self.size_min:
            return []
        if n > self.size_now():
            return random.sample(self.experience, self.size_now())
        else:
            return random.sample(self.experience, n)

    def size_now(self):
        return len(self.experience)
