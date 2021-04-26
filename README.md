# Cassava Leaf Disease Classification Kaggle Competition
This repository contains code that formed the basis of my submission to the [Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification) Kaggle competition. I finished 2616th, with a final leaderboard score of 87.54% accuracy on the test set. As this was my first Kaggle submission, I was more hoping to learn some lessons about competing, and practice implementing papers, rather than climbing the leaderboard, and in this respect I believe I succeeded, as outlined below.

## The approach
I joined the competition after it had been running for a couple of weeks, so my initial approach was to survey some of the architectures chosen by other participants, and to read some of the papers linked in the discussion. It became clear quickly that most lucrative results had been garnered by some combination of ResNet based nets and EfficientNets. As I was more familiar with Resnet from a theoretical point of view, it felt like good practice to start with a baseline based on this architecture.

I began with some very elementary EDA: I took a look at some of the images in each class, calculated the class imbalances in case oversampling during training could be utilised, and calculated normalisation constants for augmentation. I also merged the two available datasets on Kaggle (one from a previous, similar competition) to extend the training data.

I then constructed a baseline model, fine-tuning a Resnext-50 net that was pretrained on ImageNet. This is the backend I did most experimentation with. I later went on to add many model improvements, including cross-validation, early stopping, learning rate scheduling and gradient accumulation. Most of these changes made slight improvements to the model. I also implemented a number of architecture changes to try and improve performance. I list the papers I implemented or borrowed implementations of below, and discuss the final model choice in more detail.

As identified by other participants, the dataset was inherently noisy. Many images were mislabelled, or could reasonably have belonged to multiple classes. To combat this, I tried using some novel loss functions designed to smooth out label noise during training.

Bi-tempered logistic loss: https://arxiv.org/pdf/1906.03361.pdf

Taylor cross entropy loss: https://www.ijcai.org/Proceedings/2020/0305.pdf (Implementation: https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/taylor_softmax.py)

I also tried a number of different activation functions in place of the standard ReLU, mainly because I had heard on the TWIMLAI podcast that some emergent activation functions were outperforming ReLU, especially in the presence of data noise or highly periodic data. In particular, I tried the following:

Mish activation function: https://arxiv.org/vc/arxiv/papers/1908/1908.08681v1.pdf

SIREN activation function: https://arxiv.org/pdf/2006.09661.pdf

I also tried using weight standardisation as opposed to batch normalisation, as I had felt during initial experiments on my laptop that the small batch size was negatively impacting performance. However, I eventually found that gradient accumulation was a more effective solution to this problem.

Weight standardisation: https://arxiv.org/abs/1903.10520 (Implementation: https://github.com/joe-siyuan-qiao/WeightStandardization)

Finally, I decided to experiment with a novel data augmentation procedure called SnapMix. SnapMix performs mixing of training images to create new datapoints, and constructs a loss function based on the heat map of the convolutional layers in the network to capture the semantically relevant parts of the new image.

SnapMix: https://arxiv.org/abs/2012.04846 (Implementation loosely adapted from: https://www.kaggle.com/sachinprabhu/pytorch-resnet50-snapmix-train-pipeline)

## Final model
After some experimentation, the final model was an ensemble of a 5-fold CV EfficientNet B3 with a 3-fold CV SE-Resnext-50, with bi-tempered logistic loss, Mish activation functions and the Ranger optimiser, which is a combination of RAdam with Lookahead, implemented here https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer. I chose the EfficientNet-B3 because I noticed no improvements to accuracy with a larger architecture, and I chose the Squeeze and Excitation Resnet because it gave a very slight increase in accuracy over Resnext.

## Lessons learned
I had a great time implementing some new papers and trying to improve my model on the cassava leaf dataset, and it was certainly nice to work on a problem with a bit of  potential for humanitarian and environmental impact. There are, however, many things I would do differently in future competitions. Firsly, I did not leave myself enough time. I had initially hoped to train a 5 CV Resnet as well as the EfficientNet, and then do pseudo-labelling on a further dataset of 10,000 unlabelled images provided by the organisers, but I simply did not have time to do all the training before the end of the competition. I think in this respect, formulating an initial plan for a good model and testing it in the Kaggle environment rather than simply on my laptop would be a good way to go next time. I also learned how important it is to keep comprehensive records of the model performance and to make changes in small, isolated iterations. At many points, I simply added a lot of features that I thought would probably make the model better, and if they did, then that new model became my new baseline to beat. This made it difficult to isolate precisely which changes were causing the model to improve, which in turn made further fine-tuning more difficult. In general, I learned how it pays to be more organised when designing ML models, something I hope to carry forward into future competitions and future work.
