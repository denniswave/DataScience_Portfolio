# Data Science Portfolio

This readers guide provides a guide to the notebooks I've worked on during the minor. The guide will be (mostly) in chronological order, starting with the first steps into data science and working through multiple phases in the minor project.

## Preprocessing

To practice with python and pandas, we got a dataset from TNO to practice on. This was a public dataset found on Kaggle and consisted of Agora marketplace data. The [first notebook](./Dennis_van_Oosten_1_Preprocessing.ipynb) shows the preprocessing script I created after trying out a lot of different methods. This script will help easily choose what preprocessing methods we want to use in the future of the project.

## Vectorization and visualization

To make a first attempt in vectorization I tried applying Bag of Words to our data. See [notebook 2](./Dennis_van_Oosten_2_Bag_of_Words.ipynb).

To get a better understanding of how our data was structured, I tried visualizing it in [notebook 3](./Dennis_van_Oosten_3_Data_Visualization.ipynb).

Later, I tried a different vectorization method: word2vec, and visualized the vector space using t-SNE. See [notebook 6](./Dennis_van_Oosten_6_Word2Vec_&_t-SNE.ipynb).

## Machine learning

For the first attempt at training a model, I tried several on the dataset and compared the scores. See [notebook 4](./Dennis_van_Oosten_4_Training_multiple_models.ipynb).

The best result from this initial attempt was a LinearSVC model. In [notebook 5](./Dennis_van_Oosten_5_LinearSVC.ipynb) I took a more in-depth look into LinearSVC.

Later on, I wanted to take a closer look into the comparison scores for different algorithms. In [notebook 7.1](./Dennis_van_Oosten_7.1_ML_Comparisons.ipynb). I compare different algorithms against the entire dataset and in [notebook 7.2](./Dennis_van_Oosten_7.2_ML_Comparisons.ipynb) I do the same with the dataset balanced to see if that made a difference.

In [notebook 8](./Dennis_van_Oosten_7.1_ML_Comparisons.ipynb) I attempted to improve the best result of LinearSVC by tweaking the tf-idf vectorization.

## Creating reusable dataframes

To be able to work more easily, I created and saved a few different reusable dataframes where the Agora dataset was filtered in a few different ways. This way, we could easily choose which set we wanted to try out. In combination with the preprocessing script this gives the opportunity to switch between different combinations of preprocessing and balancing the data. The different dataframes created in [notebook 9](./Dennis_van_Oosten_9_Creating_Reusable_DataFrames.ipynb) are:
- All categories
- Main categories
- Balanced set (500 records per category)
- Balanced set with only main categories

## More machine learning

After a while, a few more thing were tried in an attempt to imporve the LinearSVC model, although with little success. See [notebook 10: One-vs-All Classifier](./Dennis_van_Oosten_10_One_vs_All_Classifier.ipynb) and [notebook 11: K-means sentence clustering](./Dennis_van_Oosten_11_K-Means_Sentence_Clustering.ipynb).

## Visualizing learning process

In the following notebooks, I tried to get more insight in the learning process to maybe find a way to tweak and improve the best model so far (still LinearSVC with tf-idf vectorization). See [notebook 12.1](./Dennis_van_Oosten_12.1_Learning_Curves.ipynb), [notebook 12.2](./Dennis_van_Oosten_12.2_Learning_Curves.ipynb) and [notebook 13](./Dennis_van_Oosten_13_Validation_Curves.ipynb).

## Validating on other dataset

To see if our pipeline of preprocessing, vectorization and training worked well in general, I tested it out with other datasets found on Kaggle. I compared the results from the Kaggle scoreboard to our results and they turned out quite well.
- The [IMDB reviews](./Dennis_van_Oosten_15_IMDB_Reviews.ipynb) scored 90% while the best Kaggle score was 93%.
- The [Video game comments](./Dennis_van_Oosten_16_Video_Game_Comments.ipynb) was off by 1.25 points while the Kaggle model was off by 1.22.
- The [Toxic comment](./Dennis_van_Oosten_17_Toxic_Comment.ipynb) scored 92%.
Overall, we can say that our model performs pretty well on other datasets.

## Neural networks

I also attempted to train a neural network. I did this with SKLearn which turned out to be a bad way to do it since it did not use the GPU. However, I did manage to train some models and the results weren't bad. However, they could not match LinearSVC on our dataset. See [notebook 18](./Dennis_van_Oosten_18_CNN.ipynb), [notebook 19.1](./Dennis_van_Oosten_19.1_MLP.ipynb), [notebook 19.2](./Dennis_van_Oosten_19.2_MLP.ipynb) and [notebook 19.3](./Dennis_van_Oosten_19.3_MLP.ipynb).
To compare how these neural networks compared to our best model, I trained our best model on different datasets in [notebook 21](./Dennis_van_Oosten_21_Comparing_Best_Results.ipynb).

## Topic modelling and new data

Because TNO wanted the categories mapped to the Interpol topics list and maybe also detect new topics, I tried a form of topic modelling in [notebook 20](./Dennis_van_Oosten_20_Extracting_Topics.ipynb). However, eventually we mapped the topics ourselves manually, since this wouldn't take that long and would assure that we got the best labels for further training.

To be able to combine the Agora data from Kaggle and the In [notebook 22](./Dennis_van_Oosten_22_Mapped_Dataset.ipynb) I trained the mapped Agora dataset to see how it would perform when labelled differently. In [notebook 23](./Dennis_van_Oosten_23_Mapping.ipynb) I mapped the 'darkweb markets dataset' (the new data from webIq that TNO wanted us to work on) and in [notebook 24](./Dennis_van_Oosten_24_New_Dataset.ipynb) I visualized and trained this new, mapped dataset to see how it compared to Agora.

## Balancing differently

Another method we could still try to improve our model was to balance it in a different way. The only balancing I'd tried so far was to remove entires from larger categories to make all categories of equal size and remove the ones that were too small (see 'Creating reusable dataframes'). This worked, but meant that a lot of data was lost. In [notebook 26.1](./Dennis_van_Oosten_26.1_Sample_Balancing_TFIDF.ipynb) and [notebook 26.2](./Dennis_van_Oosten_26.2_Sample_Balancing_W2V.ipynb) I tried balancing by copying records from categories that had less records to make them all of equal length. This turned out to work as well, but not better compared to the unbalanced set: [notebook 26.3](./Dennis_van_Oosten_26.3_Main_Categories_W2V.ipynb). 









