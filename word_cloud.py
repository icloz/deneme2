# Start with loading all necessary libraries
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import json
import glob
import matplotlib.pyplot as plt

#to display all columns of a df
pd.set_option("display.max_colwidth", None) 


dfs = [] # an empty list cleato store the data frames
for file in glob.glob("./ado-slack/*"):
#     print(file)
    f = open(file) # read data frame from json file
    data = json.load(f)
    data = pd.json_normalize(data,max_level=3) #flatten the nested json
    dfs.append(data) # append the data frame to the list

bigdata = pd.concat(dfs, ignore_index=True) # concatenate all the data frames in the list.
bigdata = bigdata[bigdata.parent_user_id.notnull() == False]#get the parent message in a thread 

bigdata_cloud = bigdata.rename(columns = {'user_profile.real_name': 'User_wCloud'}, inplace = False)    
bigdata_cloud['User_wCloud'] = bigdata_cloud['User_wCloud'].str.replace(" ","")

print(bigdata_cloud[["User_wCloud"]].head(20)) 
User_wCloud = bigdata_cloud.groupby("User_wCloud")
# print(User_wCloud.describe().head())
print(User_wCloud.count().sort_values(by="reply_count",ascending=False).head())




# Python program to generate WordCloud
comment_words =''
# stopwords = set(STOPWORDS)
# print('0',stopwords)

stopwords = {'NaN','thank','tomtomgroup','will','team','work','hi','using','use','thanks','hello','see','one','need',
             'her', 'was', "how's", 'just', 'else', 'whom', 'yourself', 'which', 'ourselves', 'further', 'at', 'through', "they'll", 'get', 'under', "we'd", 'being', 'below', "why's", 'between', "here's", 'not', "what's", 'when', "there's", "we'll", 'me', 'some', 'own', 'www', 'am', 'also', "i'm", "you'd", 'in', 'such', "they'd", "i'd", 'them', 'myself', "doesn't", "shouldn't", "you're", 'she', 'him', "she'd", "we've", 'i', 'other', 'we', 'against', "she's", 'having', 'their', 'each', 'how', "we're", 'a', 'cannot', "isn't", "won't", 'com', 'because', 'yours', 'any', "let's", 'hers', 'but', 'what', "don't", 'itself', 'like', 'they', "hadn't", 'its', "didn't", 'nor', 'off', 'once', 'theirs', 'all', 'he', 'herself', 'down', "that's", 'with', 'if', "he's", "weren't", "can't", 'about', "i've", 'doing', 'and', 'is', "she'll", 'over', 'too', "he'll", 'ought', "they've", 'who', 'on', 'from', 'hence', 'do', 'here', 'that', 'yourselves', 'his', 'then', 'no', 'would', "shan't", 'been', 'for', 'few', 'http', 'your', 'into', 'should', 'of', 'himself', "he'd", "haven't", 'can', 'has', 'while', 'these', 'above', 'same', 'most', 'those', 'have', 'otherwise', 'the', 'shall', 'could', 'both', 'ours', "wasn't", 'an', 'where', 'k', 'until', 'this', "they're", 'does', 'only', 'r', 'our', 'after', "wouldn't", 'my', 'so', "when's", 'as', 'by', 'be', 'there', 'it', "aren't", 'therefore', 'since', 'did', 'before', 'up', 'very', "couldn't", 'than', 'to', "you'll", "you've", 'more', 'or', 'are', 'out', "mustn't", 'why', 'you', "who's", 'however', "hasn't", "where's", 'again', "i'll", 'ever', 'themselves', 'during', 'were', "it's", 'had'}

# for wordcloud based on messages
# for val in bigdata_cloud.text: 
# for wordcloud based on message sender
for val in bigdata_cloud.User_wCloud:
    
    # typecaste each val to string
    val = str(val)

    # split the value
    tokens = val.split()
    
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    
    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)

# plot the WordCloud image					
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


print("There are {} observations and {} features in this dataset. \n".format(bigdata_cloud.shape[0],bigdata_cloud.shape[1]))
print("There are {} types of users in this dataset such as {}... \n".format(len(bigdata_cloud.user.unique()),", ".join(bigdata_cloud.User_wCloud.unique()[0:5])))
# print("There are {} teams asking questions in this dataset such as {}... \n".format(len(bigdata.team.unique()),", ".join(bigdata.team.unique()[0:5])))