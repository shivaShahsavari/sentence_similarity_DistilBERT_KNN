# Kaggle_competition

This challenge was proposed by ...

To get better insight within data, I had three EDAs. EDAs are really necessary to check which approaches can be proper for next steps: (EDA.py)  
## EDA  
### 1) Analysis 1  
In first analysis, idea of topic classification using CNN with different kernels is explored. So, you can see in EDA file that min and max number of words in data sources name are 1 and 17. Unfortunately, Kernel lengths in range of 1-17 needs 62 GB RAM to run on training dataset.  
**result:  
max length of label: 17  
max_label: "National Center for Science and Engineering Statistics Survey of Graduate Students and Postdoctorates in Science and Engineering  
min length of label: 1  
min_label: ADNI**  

### 2) Analysis 2  
Since text of articles are too much, text summarization idea came to my mind. So, I checked how many titles of articles including data source names. If their numbers are a few, I could get rid of rest parts and focus only on text with mentioned titles. You can see top 25 articles including data sources name. 

<img src="image/EDA2.png" width="900" height="500">  

Total distinct titles of articles are 12828 and if we chose top 30 articles, around 10064 of 19662 articles will not be involved in modeling. By further look at data, 9934 files will be missed considering 40 top articles. So, this approach is not suitable as well.  
**result:
number of files not in top 30 titles:  10064  
number of files not in top 40 titles:  9938**  

## 3) Analysis 3  

3) Using sentenceTransform & indexing
