import random
from collections import defaultdict

class Modulator():
    def __init__(self, noise_list):
        self.noise = noise_list
        self.weights = [1.0/len(noise_list)] * len(noise_list)
        
    def modulate(self, evalute_results):
        total = sum([e if e > 0.01 else 0.01 for e in evalute_results])
        weights = [e/total if e > 0.01 else 0.01/total for e in evalute_results]
        self.weights = weights

    def chose_noise(self):
        selected = random.choices(self.noise,weights=self.weights, k=1)[0]
        return selected
    
if __name__=="__main__":
    noise_modulator = Modulator(['Identity','Crop', 'Cropout', 'Dropout', 'Resize', 'GaussianBlur',  'GaussianNoise', 'Brightness', 'Contrast', 'Saturation','VAERecon','Jpeg','Rotation','Flip'])
    for i in range(10):
        results = [random.uniform(0.5, 1.0) for _ in range(14)]
        print(f"Before Modulate {noise_modulator.weights}")
        print(f"Noise Choice {noise_modulator.chose_noise()}")
        noise_modulator.modulate(results)
        print(f"After Modulate {noise_modulator.weights}")
