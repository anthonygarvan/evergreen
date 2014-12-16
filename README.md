### Evergreen Challenge Submission


This is my submission to Kaggle's evergreen challenge, which challenged
participants to classify StumbleUpon webpages between those that had enduring 
interest for its users, which they call evergreen, and those that were ephermal 
\- ie., had little value to recommend a few days or weeks after being written.

One interesting finding from this work was that recipes for holidays were typically
evergreen, with the exception of super bowl recipes, which were consistently ephemeral.

I tried many things here, including random forests, logistic regression, naive bayes, 
and I also used a pre-trained word2vec deep learning tool to try to make the rare words
less sparse by collecting them into more generic vectors.