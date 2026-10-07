1. What is a tensor? Difference with numpy array.
2. Structure of neural network (breakdown)
3. Non-determinism
4. Normalization view: min-max normalization (ToTensor), Standard-score normalization, centers the data; motivate normalization, the important thing is that both training and testing data need to be normalized
5. Example tensor shape error
6. CPU training and loss curves: learning rate 0.05, seed 0, batch size 64
7. Vary the seed (i.e., seed = 1) -> pretty stable overall
8. Vary the learning rate: from 0.05 to 0.1
9. GPU bug
10. Execute the training on GPU, you'll see that the performance of the unnormalized version can vary quite a lot, while the one of the normalized version stays roughly the same
11. Exercise for you: what happens if I increase the batch size from 64 to 1024? Next time you can tell me what you got and why.