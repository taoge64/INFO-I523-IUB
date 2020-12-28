# Big Data Application in E-commerce

[![Check Report](https://github.com/cybertraining-dsc/fa20-523-339/workflows/Check%20Report/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-339/actions)
[![Status](https://github.com/cybertraining-dsc/fa20-523-339/workflows/Status/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-339/actions)
Status: final, Type: Project

Tao Liu, [fa20-523-339](https://github.com/cybertraining-dsc/fa20-523-339/), [Edit](https://github.com/cybertraining-dsc/fa20-523-339/blob/main/project/project.md)


{{% pageinfo %}}

## Abstract

As a result of the last twenty year's Internet development globally, the E-commerce industry is getting stronger and stronger. While customers enjoyed their convenient online purchase environment, E-commerce sees the potential for the data and information customers left during their online shopping process. One fundamental usage for this information is to perform a Recommendation Strategy to give customers potential products they would also like to purchase.  This report will build a User-Based Collaborative Filtering strategy to provide customer recommendation products based on the database of previous customer purchase records. This report will start with an overview of the background and explain the dataset it chose *Amazon Review Data*. After that, each step for the code and step made in a corresponding file *Big_tata_Application_in_E_commense.ipynb* [^6] will be illustrated, and the User-Based Collaborative Filtering strategy will be presented step by step.

Contents

{{< table_of_contents >}}

{{% /pageinfo %}}

**Keywords:** recommendation strategy,user-based, collaborative filtering, business, big data, E-commerce, customer behavior 

## 1. Introduction

Big data have many applications in scientific research and business, from those in the hardware perspective like Higgs Discovery to the software perspective like E-commence. However, with the passage of time, online shopping and E-commerce have become one of the most popular events for citizens' lives and society. Millions of goods are now sold online the customers all over the world. With the 5G technology's implementation, this trend is now inevitable. These activities will create millions of data about customer's behaviors like how they value products, how they purchase or sell the products, and how they review the goods purchased would have a tremendous contribution for corporations to analyze. These data can not only help convince the current strategies of E-commerce on the right track, but a potential way to see which step E-commerce can make up for attracting more customers to buy the goods. At the same time, these data can also be implemented as a way for recommendation strategies for E-commerce. It will help customers find the next products they like in a short period by implementing machine learning technology on Big Data. The corporations also can enjoy the increase of sales and attractions by recommendation strategies. A better recommendation strategy on E-commerce is now the new trend for massive data scientists and researchers’ target. Therefore, this field is now one of the most popular research areas in the data science fields. 

In this final project, An User-Based Collaborative Filtering Strategy will be implemented to get a taste of the recommendation strategy based on Customer's Gift card purchase records and the item they also viewed and bought. The algorithm's logic is the following: A used record indicates that customer who bought product A and also view/buy products B&C. When a new customer comes and shows his interest in B&C, product A would be recommended. This logic is addressed based on the daily-experience of customer behaviors on their E-commerce experience. 

## 2. Background

Recommendation Strategy is a quite popular research area in recent years with a strong real-world influence. It is largely used in E-commerce platforms like Taobao, Amazon, etc. Therefore, It is obvious that there are plenty of recommendation strategies have been done. Though every E-commerce recommendation algorithm may be different from each other, the most popular technique for recommendation systems is called Collaborative Filtering. It is a technique that can filter out items that a user might like based on reactions by similar users. During this technique, the memory-based method is considered in this report since it uses a dataset to calculate the prediction using statistical techniques. This strategy will be able to fulfill in the local environment with a proper dataset. There are two kinds of memory-based methods available in the market: *User-Based Collaborative Filtering*, *Item-Based Collaborative Filtering* [^1]. This project will only focus on the User-Based Collaborative Filtering Strategy since Item-Based Collaborative Filtering requires a customer review rate for evaluation. The customer review rate for evaluation is not in the dataset available in the market. Therefore, Item-Based Collaborative Filtering unlikely to be implemented, and the User-Based Collaborative Filtering Strategy is considered. 

## 3. Choice of Data-sets


The dataset for this study is called *Amazon Review Data* [^2]. Particularly, since the dataset is now reached billions of amount, the subcategory gift card will be used as an example since the overall customer record is 1547 and the amount of data retrieved is currently in the right amount of training. This fact can help to perform User-Based Collaborative Filtering in a controlled timeline.

## 4. Data Preprocessing and cleaning


The first step will be data collection and data cleaning. The raw data-set is imported directly from data-set contributors' online storage *meta_Gift_Cards.json.gz* [^3] to Google Colab notebook. The raw database retrieved directly from the website will be shown in **Table 1**.


|  Attribute   | Description                    |  Example                                |
| :-----------:| :-----------------------------:|:---------------------------------------:|
|  category    | The category of the record     | \[\"Gift Cards", "Gift Cards"\]\        |
|  tech1       | tech relate to it              | ""                                      |
|  description | The description of the product |"Gift card for the purchase of goods..." |
|  fit         | fit for its record             |""                                       |
|  title       | title for the product          |"Serendipity 3 $100.00 Gift Card"        |
| **also_buy** | the product also bought        |\[\"B005ESMEBQ"\]\                       |
|  image       | image of the gift card         |""                                       |
|  tech2       | tech relate to it              |""                                       |
|  brand       | brand of the product           |"Amazon"                                 |
|  feature     | feature of the product         |"Amazon.com Gift cards never expire"     |
|  rank        | rank of the product            |""                                       |
| **also_view**| the product also view          |\[\"BT00DC6QU4"\]\                       |
|  details     | detail for the product         |"3.4 x 2.1 inches ; 1.44 ounces"         |
|  main_cat    | main category of the product   |"Grocery"                                |
|  similar_item| similar_item of the product    |""                                       |
|  date        | date of the product assigned   |""                                       |
|  price       | price of the product           |""                                       |
|  **asin**    | product asin code              |"B001BKEWF2"                             |

**Table 1:** The description for the dataset

Since the attributes *category*, *main_cat* are the same for the whole dataset, they will not be valid training labels. The attributes *tech1*, *fit*, *tech2*, *rank*, *similar_item*, *date*, *price*  have no/ extremely less filled in. That made them also invalid for being training labels. The attributes *image*, *description* and *feature* is unique per item and hard to find the similarity in numeric purpose and then hard to be used as labels. Therefore, only attributes **also_buy**,  **also_view**, **asin** are trained as attributes and labels in this algorithm. **Figure 1** is a shortcut for the raw database.
```
THE RAW DATABASE 
The size of DATABASE : 1547
                                                      0
0     {"category": ["Gift Cards", "Gift Cards"], "te...
1     {"category": ["Gift Cards", "Gift Cards"], "te...
2     {"category": ["Gift Cards", "Gift Cards"], "te...
3     {"category": ["Gift Cards", "Gift Cards"], "te...
4     {"category": ["Gift Cards", "Gift Cards"], "te...
...                                                 ...
1542  {"category": ["Gift Cards", "Gift Cards"], "te...
1543  {"category": ["Gift Cards", "Gift Cards"], "te...
1544  {"category": ["Gift Cards", "Gift Cards"], "te...
1545  {"category": ["Gift Cards", "Gift Cards"], "te...
1546  {"category": ["Gift Cards", "Gift Cards"], "te...

[1547 rows x 1 columns]
```
**Figure 1:** The raw database

For the training purpose, all asins that appeared in the dataset, either from *also_buy & also_view* list or * asin*, have to be reformatted from alphabet character to numeric character. For example, the original label for a particular item may be called **B001BKEWF2**. It will now be reformatted to a numeric number as 0. In that case, it can be a better fit-in the next step training method and easy to track. This step will be essential since it will help the also_view and also_buy dataset to be reformatted and make sure they are reformed in the track without overlapping each other. Therefore, a reformat_asin function is called for reformatting all the asins in the dataset and is performed as a dictionary. A shortcut for the *Asin Dictionary* is shown in **Figure 2**.

```
The 4561  Lines of Reformatted ASIN reference dictionary as following.
{'B001BKEWF2': 0, 'B001GXRQW0': 1, 'B001H53QE4': 2, 'B001H53QEO': 3, 'B001KMWN2K': 4, 'B001M1UVQO': 5, 
 'B001M1UVZA': 6, 'B001M5GKHE': 7, 'B002BSHDJK': 8, 'B002DN7XS4': 9, 'B002H9RN0C': 10, 'B002MS7BPA': 11, 
 'B002NZXF9S': 12, 'B002O018DM': 13, 'B002O0536U': 14, 'B002OOBESC': 15, 'B002PY04EG': 16, 'B002QFXC7U': 17, 
 'B002QTM0Y2': 18, 'B002QTPZUI': 19, 'B002SC9DRO': 20, 'B002UKLD7M': 21, 'B002VFYGC0': 22, 'B002VG4AR0': 23, 
 'B002VG4BRO': 24, 'B002W8YL6W': 25, 'B002XNLC04': 26, 'B002XNOVDE': 27, 'B002YEWXZ0': 28, 'B002YEWXMI': 29, 
 'B003755QI6': 30, 'B003CMYYGY': 31, 'B003NALDC8': 32, 'B003XNIBTS': 33, 'B003ZYIKDM': 34, 'B00414Y7Y6': 35, 
 'B0046IIHMK': 36, 'B004BVCHDC': 37, 'B004CG61UQ': 38, 'B004CZRZKW': 39, 'B004D01QJ2': 40, 'B004KNWWPE': 41, 
 'B004KNWWP4': 42, 'B004KNWWR2': 43, 'B004KNWWRC': 44, 'B004KNWWT0': 45, 'B004KNWWRW': 46, 'B004KNWWQ8': 47, 
 'B004KNWWNG': 48, 'B004KNWWPO': 49, 'B004KNWWXQ': 50, 'B004KNWWUE': 51, 'B004KNWWYU': 52, 'B004KNWWWC': 53, 
 'B004KNWX3A': 54, 'B004KNWX1W': 55, 'B004KNWWZE': 56, 'B004KNWWSQ': 57, 'B004KNWX4Y': 58, 'B004KNWX12': 59, 
 'B004KNWX3U': 60, 'B004KNWX62': 61, 'B004KNWX2Q': 62, 'B004KNWX6C': 63...}
```
**Figure 2:** The ASIN dictionary

Then the data contained in the each record's attributes: **also_view** & **also_buy** will be reformated as **Figure 3** and **Figure 4**. **Figure 3** is about the also_view item in reformatted numeric numbers based on each item customer purchased. **Figure 4** is about the also_buy item in reformatted numeric numbers based on each item customer purchased.

```
also_view List: The first 10 lines
Item  0 :  []
Item  1 :  [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 
            2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
Item  2 :  [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 922, 2036, 283, 
            2037, 2038, 2001, 2000, 2013, 2039, 2040, 2007, 2041, 2042, 2009, 1233, 2043, 
            2014, 234, 2044, 2012, 2005, 2045, 2046, 2002, 2047, 378, 2048, 1382, 2008, 
            2004, 2011, 2049, 2050, 2051, 2052, 2003, 2053, 2054, 2018, 2055, 2056]
Item  3 :  []
Item  4 :  []
Item  5 :  []
Item  6 :  []
Item  7 :  []
Item  8 :  []
Item  9 :  []
Item  10 :  [2057, 2058, 2059]
```
**Figure 3:** The also_view list

```
also_buy List: The first 20 lines
Item  0 :  []
Item  1 :  []
Item  2 :  [2026, 2028, 2027, 2049, 1382, 2037, 2012, 2023]
Item  3 :  []
Item  4 :  []
Item  5 :  []
Item  6 :  []
Item  7 :  []
Item  8 :  [4224, 4225, 4226, 4227, 4228, 4229, 4230, 4231, 4232, 4233, 
            4234, 4235, 4236, 4237, 4238, 4239, 4240, 4241, 4242, 4243, 
            4244, 4245, 4246, 4247, 4248, 4249, 4250, 4251, 4252]
Item  9 :  []
Item  10 :  []
```
**Figure 4:** The also_buy list

While the also_buy list and also_view list is addressed. It is also important to know how many times a particular item appeared in other items' also view list and also buy list. These dictionaries will help to calculate the recommendation rate later.  **Figure 5** and **Figure 6** is an example for how many times item 2000 appeared in other item's also_view and also_buy lists.
```
also_view dictionary: use Item 2000 as an example
Item  2000 :  [1, 2, 11, 12, 51, 60, 63, 65, 66, 67, 85, 86, 90, 94, 99, 100, 101, 103, 107, 108, 113, 116, 123, 126, 127, 129, 130, 141, 142, 143, 145, 146, 147, 148, 194, 199, 200, 204, 217, 221, 225, 229, 230, 231, 232, 233, 234, 235, 251, 253, 254, 260, 264, 268, 269, 270, 271, 280, 284, 285, 286, 287, 288, 294, 295, 296, 298, 299, 305, 306, 307, 308, 309, 313, 319, 327, 328, 338, 339, 344, 346, 348, 355, 356, 360, 371, 372, 377, 380, 389, 394, 406, 407, 410, 415, 440, 456, 469, 480, 490, 494, 495, 496, 502, 505, 509, 511, 512, 514, 517, 519, 520, 527, 530, 548, 591, 595, 600, 608, 609, 621, 631, 633, 670, 671, 672, 673, 675, 681, 689, 691, 695, 697, 707, 708, 709, 719, 783, 792, 793, 796, 797, 801, 803, 804, 807, 810, 816, 817, 818, 819, 836, 840, 842, 856, 892, 902, 913, 914, 917, 921, 955, 968, 972, 974, 975, 979, 981, 990, 991, 997, 998, 999, 1000, 1001, 1003, 1005, 1006, 1007, 1010, 1011, 1014, 1015, 1017, 1018, 1023, 1024, 1026, 1027, 1028, 1031, 1032, 1035, 1037, 1038, 1039, 1040, 1042, 1043, 1050, 1069, 1070, 1084, 1114, 1115, 1116, 1117, 1119, 1143, 1153, 1171, 1175, 1192, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1207, 1208, 1213, 1217, 1218, 1220, 1222, 1233, 1236, 1238, 1242, 1244, 1245, 1246, 1249, 1251, 1258, 1268, 1270, 1280, 1285, 1289, 1290, 1292, 1295, 1315, 1318, 1319, 1324, 1328, 1330, 1333, 1336, 1341, 1345, 1346, 1347, 1348, 1352, 1359, 1361, 1365, 1366, 1373, 1378, 1384, 1389, 1394, 1395, 1396, 1403, 1405, 1406, 1407, 1414, 1415, 1417, 1418, 1419, 1420, 1423, 1424, 1426, 1427, 1430, 1431, 1432, 1433, 1434, 1437, 1443, 1453, 1454, 1455, 1457, 1458, 1462, 1463, 1464, 1467, 1468, 1469, 1470, 1472, 1474, 1475, 1477, 1478, 1480, 1481, 1482, 1486, 1488, 1492, 1496, 1497, 1498, 1499, 1500, 1501, 1504, 1505, 1506, 1508, 1509, 1512, 1513, 1514, 1515, 1523, 1530, 1533, 1537, 1539, 1546]
```
**Figure 5:** The also_view dictionary

```
also_buy dictionary: use Item 2000 as an example
Item  2000 :  [217, 231, 235, 236, 277, 284, 285, 286, 287, 306, 307, 308, 327, 
               359, 476, 482, 505, 583, 609, 719, 891, 922, 963, 1065, 1328, 1359, 
               1384, 1399, 1482, 1483, 1490, 1496, 1497, 1499, 1509, 1512, 1540]
```
**Figure 6:** The also_buy dictionary

## 5. Recommendation Rate and Similarity Calculation

While all the dictionaries and attributes-label relationship are prepared in Part4, the recommendation rate calculation is addressed in this part. There are two types of similarity methods in this algorithm: **Cosine Similarity** and **Euclidean Distance Similarity** that perform the similarity calculation. Before calculating the similarity, the first step would be phrasing the recommendation rate for each item to another item. The **Figure 8** is a shortcut for the recommendation rate matrix. It will use the logic in **Figure 7**.

```
for item in the asin list:
  for asin in the also_view dictionary:
    if asin is founded in also_view dictionary[item] list:
        score for this item increase 2
    each item in the also_view_dict[asin]'s score will be also increase 2
  for asin in the also_view dictionary:
    if asin is founded in also_view dictionary[item] list:
        score for this item increase 10
    each item in the also_view_dict[asin]'s score will be also increase 10
for other scores which is currently 0, assigned the average value for it
return the overall matrix for the further step
```
**Figure 7:** The sudocode for giving the recommendation rate for the likelyhood of the next purchase item based the current purchase

```
Item  0 :  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ...]
Item  1 :  [13.0, 52, 28, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 2, 2, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0 ...]
Item  2 :  [29.5, 28, 182, 29.5, 29.5, 29.5, 29.5, 29.5, 29.5, 29.5, 29.5, 4, 2, 29.5, 29.5, 29.5, 29.5, 29.5, 29.5, 29.5 ...]
Item  3 :  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ...]
Item  4 :  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ...]
Item  5 :  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ...]
Item  6 :  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ...]
Item  7 :  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ...]
Item  8 :  [14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 290, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5 ...]
Item  9 :  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ...]
Item  10 :  [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 6, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 ...]
```
**Figure 8:** The shortcut for recommenation rate matrix

The first similarity method implemented is *Cosine Similarity* [^4]. It will use the cosine of the angle between vectors(see **Figure 9**) to address the similarity between different items. By implementing this method with *sklearn.metrics.pairwise* package, it will rephrase the whole recommendation similarity as **table 2**. 

![image info](https://github.com/cybertraining-dsc/fa20-523-339/raw/main/project/images/cosine-similarity.png)

**Figure 9:** The cosine similarity

|item|0  |1. |2        |3  |4  |5  |6  |7. |8.      |...1547 |
|:-: |:-:|:-:|:-:      |:-:|:-:|:-:|:-:|:-:|:-:     |:-:     |
|0   |0. |0. |0.       |0. |0. |0. |0. |0. |0.      | ...    |	
|1   |0. |0. |0.928569 |0. |0. |0. |0. |0. |0.873242|...     |
|2   |0. |0. |0.928569 |0. |0. |0. |0. |0. |0.      | ...    |

**table 2:** The shortcut for using consine similarity to address the recommendation result

The second similarity method implemented is *Euclidean Distance Similarity*[^5]. It will use Euclidean Distance to calculate the distance between each items as a way to calculate similarity (see **Figure 10**). By implementing this method with *scipy.spatial.distance_matrix* package, it will rephrase the whole recommendation similarity. **Figure 11** is an example with the item 1.

![image info](https://github.com/cybertraining-dsc/fa20-523-339/raw/main/project/images/euclidean.png)

**Figure 10** The Euclidean Distance calculation

```
Item  1 :  [1005.70671669    0.         1370.09142031 1005.70671669 1005.70671669
 1005.70671669 1005.70671669 1005.70671669  710.89169358 1005.70671669
  905.23339532  862.88933242  971.0745337  1005.70671669 1005.70671669
 1005.70671669 1005.70671669 1005.70671669 1005.70671669 1005.70671669...]
```
**Figure 11:** The Euclidean Distance similarity example of item 1

## 6. Accuracy

The accuracy for the consine_similarity and euclidean distance similarity with the number of items already purchased is shown as **Figure 12**. Here the blue line represented the cosine similarity, and the red line represented the euclidean distance similarity. As presented, the more item joined as purchased, the less likely both similarity algorithms accurately locate the next item the customer may want to purchase next. However, overall the consine_similarity performed better accuracy compared to Euclidean Distance similarity. In **Figure 13**, both accurate number and both wrong number is addressed. Both wrong numbers changed dramatically after more already-purchased items joined. This fact convinces the prior statement: this algorithm works better when the given object is *1* but can't handle many purchased item.

![image info](https://github.com/cybertraining-dsc/fa20-523-339/raw/main/project/images/CosVSEuc.png)

**Figure 12:** The Cosine similarity and Euclidean Distance Accuracy Comparison

![image info](https://github.com/cybertraining-dsc/fa20-523-339/raw/main/project/images/bothrightandwrong.png)

**Figure 13:** The bothright and bothwrong accuracy comparison

## 7. Benchmark

The Benchmark for each step for the project is stated in **Figure 14** The overall Time spent is affordable. The accuracy calculation part(57s) and the Euclidean Distance algorithm implementation(74s) have taken the majority of time for the running. The Accuracy time consumed would be considered proper since it will randomly assign one to ten items and perform recommendation items based on it. The time spent is necessary and should be considered normal. The Euclidean Distance algorithm would be considered making sense since it is trying to perform the difference in two 1547X1547 matrixs. 

|Name                                                 | Status   |   Time | 
|:-:                                                  |:-:       |:-:     |
| Data Online Downloading Process                     | ok       |  1.028 |
| Raw Database                                        | ok       |  0.529 |
| Database Reformatting process                       | ok       |  0.587 |
| Recommendation Rate Calculation                     | ok       |  1.197 |
| Consine_Similarity                                  | ok       |  0.835 |
| Euclidean distance                                  | ok       | 73.895 |
| Recommendation strategy showcase-Cosine_Similarity  | ok       |  0.003 |
| Recommendation strategy showcase-Euclidean distance | ok       |  0.004 |
| Accuracy                                            | ok       | 57.119 |
| Showcase-Cosine_Similarity                          | ok       |  0.003 |
| Showcase-Euclidean distance                         | ok       |  0.003 |


**Figure 14:** Benchmark

The time comparison for Cosine Similarity and Euclidean Distance Time Comparison is addressed in **Figure 15** As stated, the euclidean distance algorithm has taken much more time than cosine similarity. Therefore, the cosine similarity should be considered as efficient in these two similarities.

![image info](https://github.com/cybertraining-dsc/fa20-523-339/raw/main/project/images/timecompare.png)

**Figure 15:** The Cosine Similarity and Euclidean Distance Time Comparison


## 8. Conclusion

This project *Big_tata_Application_in_E_commense.ipynb* [^6] is attempted to get a taste of the recommendation strategy based on *User-Based Collaborative Filtering*. Based on this attemption, the two similarity methods: **Cosine Similarity** and **Euclidean Distance** are addressed. After analyzing accuracy and time consumption for each method, Cosine Similarity performed better in both the accuracy and implementation time. Therefore the cosine similarity method is recommended to use in the recommendation algorithm strategies.
This project should be aware of Limitations. Since the rating attribute is missing in the dataset, the recommendation rate was assigned by the author. Therefore, in real-world implementation, both methods' accuracy can be expected to be higher than in this project. Besides, the cross-section recommendation strategies are not implemented. This project is only focused on the gift card section recommendations. With the multiple aspects of goods customer purchases addressed, both methods' accuracy can also be expected to be higher.

## 9. Acknowledgements


The author would like to thank Dr. Gregor Von Laszewski, Dr. Geoffrey Fox, and the associate instructors in the *FA20-BL-ENGR-E534-11530: Big Data Applications* course (offered in the Fall 2020 semester at Indiana University, Bloomington) for their continued assistance and suggestions with regard to exploring this idea and also for their aid with preparing the various drafts of this article.

## 10. References

[^1]: Build a Recommendation Engine With Collaborative Filtering. Ajitsaria, A. 2020 
      <https://realpython.com/build-recommendation-engine-collaborative-filtering/> 

[^2]: Justifying recommendations using distantly-labeled reviews and fined-grained aspects. Jianmo Ni, Jiacheng Li, Julian McAuley. Empirical Methods in Natural Language Processing (EMNLP), 2019 <http://jmcauley.ucsd.edu/data/amazon/> 

[^3]: meta_Gift_Cards.json.gz <http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Gift_Cards.json.gz>

[^4]: Recommendation Systems : User-based Collaborative Filtering using N Nearest Neighbors. Ashay Pathak. 2019
<https://medium.com/sfu-cspmp/recommendation-systems-user-based-collaborative-filtering-using-n-nearest-neighbors-bf7361dc24e0>


[^5]: Similarity and Distance Metrics for Data Science and Machine Learning. Gonzalo Ferreiro Volpi. 2019
<https://medium.com/dataseries/similarity-and-distance-metrics-for-data-science-and-machine-learning-e5121b3956f8>

[^6]: Big_tata_Application_in_E_commense.ipynb <https://github.com/cybertraining-dsc/fa20-523-339/raw/main/project/Big_tata_Application_in_E_commense.ipynb>


