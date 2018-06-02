Linear SVM with squared hinge loss function

This code implements a Linear SVM algorithm using a squared hinge loss function. It uses a fast gradient descent algorithm to find the values of beta that minimizes the objective function of the squared hinge loss function, which is:

F(β)=1n∑i=1n(max(0,1−yixTiβ))2+λ||β||22
 
It uses cross validation on a hold out set to determine the optimal value of the parameter lambda. It then fits a model using the optimal parameter and makes predictions.
There are other parameters you can set and their optimal values will depend on the data you are using. Examples of other parameters include:
alpha = constant used to define sufficient decrease condition within the backtracking function
gamma = fraction by which we decrease t if the previous T doesn't work, within the backtracking function
maxIter = maximum iterations for algorithms

There are three examples demonstrated here.

The file 'SVMCode-SpamDataExample.ipynb' and uses the Spam dataset which can be found here: 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data'. In this set we train our classifier to identify whether various examples of X are spam (+1) or not spam (-1).

The file 'SVMCode-SimulatedDataExample' runs the algorithms on simulated data.


The file 'SVMCode-SpamDataExample-CompareToScikit-Learn' runs the first example above and demonstrates that the algorithms created result in the same outputs as Scikit-Learn's LinearSVM function

References:
Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011
Jones, Corinne. (2018). Statistical Machine Learning For Data Scientists, Lab Materials, ipynb files. Retrieved from https://canvas.uw.edu
Harchaoui, Zaid. (2018). Statistical Machine Learning For Data Scientists, Course notes. Retrieved from https://canvas.uw.edu
