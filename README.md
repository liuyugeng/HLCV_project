# HLCV Project

This is a project based on attribute inference attack (or overlearning attack)[<sup>1</sup>](#refer1). We follow the experiment settings on CelebA dataset[<sup>2</sup>](#refer2). 

Besides the original attack, we also try this attack on Knowledge Distillation(KD) models and Differential Privacy(DP) models becuase these two defense mechanisms are the most common ones in machine learning models against some attacks such as membership inference attacks and adversarial examples.

We also test the outputs of the last 3 and 4 layer from target models in the experiments.


<div id="refer1"></div>
- [1] C.Song, V.Shmatikov; Overlearning Reveals Sensitive Attributes
<div id="refer2"></div>
- [2] Z.Liu, P.Luo, X.Wang and X.Tang; Deep Learning Face Attributes in the Wild
