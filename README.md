
In Machine Learning problems, hyperparameters are adjustable parameters that strongly influence model performance, training, and generalization. Traditionally, their selection is left to the user, which introduces a significant risk in scenarios where the search space is large and complex.

To address this, many methods for hyperparameter optimization have been developed. While some methods may provide high accuracy, no single approach consistently performs best across all datasets and problem types.

We introduce RegHyperopt, a method specifically designed for problems where evaluating the quality of a parameter set takes a long time and the target model functions as a black box (with limited transparency into its internal structure).

RegHyperopt is compared with Random Search and Bayesian Optimization, and its efficiency and accuracy are demonstrated on three benchmark datasets: CIFAR, MNIST, FASHION-MNIST.

The results show that RegHyperopt provides clear advantages over both random search and Bayesian optimization in terms of performance and efficiency.
