---
layout: post
title: How to simplify your holiday festive meal planning 
subtitle: A Recipe-Diffficulty-Tagger and the MenuPlannerHelper App to the rescue!
date: 2017-12-22 16:11:27
author: H-RM Merkle-Tan
comments: true
published: true
---

**Once again, the holiday season is upon us.** As you go about your celebrations, you may be both excited and overwhelmed by all the various gatherings and parties. Some of us will attend as guests, and perhaps a few of us might be hosting. Should you find yourself preparing the whole meal or offering to contribute a dish or two and in the mood for homemade culinary adventures, there’s a little web application, called the [*MenuPlannerHelper*](https://github.com/hengrumay/recipes) (abbreviated as [***MenuHelper***](https://bit.ly/menuplannerhelper)) I developed a while back that could come in handy.

*The inspiration for its creation come from the fact that while I generally browse for new recipes to try, I sometimes find myself in situations where there will be recipes with accompanying food photography that look amazing and seemingly ‘doable’, yet upon embarking on the preparations, they present as more challenging than originally anticipated. In those scenarios, I find myself wishing that there might be an opportunity to use the ingredients for another similar but ‘simpler’ recipe. Likewise, there are other occasions, when a recipe turns out somewhat less complicated and you might wish to add some sophistication without over-tweaking it. Having cooked with others and listened to various kitchen nightmares, I have reason to believe that my experiences are not unique.*

When festivities and inherent traditions place us under time, logistical and resource constraints, you don’t necessarily have much room to “mess up” when you are preparing a meal for others. With various things to coordinate, the last thing we really need while multi-tasking is a kitchen disaster! 

### This is where the *[MenuHelper](https://bit.ly/menuplannerhelper)* app can be useful. 
> **It was developed to help categorize recipes into different levels of complexity so that one could experience less stress and more enjoyment in coordinating the preparation of various dishes for e.g. a dinner party. Importantly, because of how it is trained to categorize recipes for relative preparation difficulty, one could further associate recipes based on the degree of similarity between their ingredients. This becomes handy if you wish to pivot between difficulty levels but also use similar ingredients that you may already have available.**

### But how does one actually classify recipes? 
**Not all recipes are created equal** – I learnt this the hard way. Recipes differ on the number of ingredients whether they are exotic or locally sourced, how ingredients are prepared, the type of cooking technique or the equipment required, and so forth. If there would be a way to classify recipes as ‘easy’ or ‘challenging’ perhaps this information could help one to better plan and prepare the combination of dishes for a festive meal. 

![](https://raw.githubusercontent.com/hengrumay/hengrumay.github.io/master/_posts/MenuPlannerHelper/NotAllRecipesCreatedEqual.png)<center>FIG1: <i>The many ways that make each recipe different</i></center>

Interestingly, as I sifted through various sources of recipes the information about its preparation difficulty is not always available, nor is it often explicitly stated. It is possible that how difficult a recipe is may depend on your cooking experience, and it might be best left to self-discovery. Yet basic and advanced cooking classes exist, so it might be worth asking:  
> ### Could we learn from available recipes that are already categorized for their relative difficulty, which aspects or features contribute to their preparation ease or complexity? 

<br>

### The ingredients and steps involved:

![](https://raw.githubusercontent.com/hengrumay/hengrumay.github.io/master/_posts/MenuPlannerHelper/CreateNUseRecipeDiffTagger.png)<center>FIG2: <i>A visual summary of the steps taken to create and use a Recipe-Difficulty-Tagger</i></center>
<br>



### Details of **i) building a Recipe-Difficulty-Tagger** and **ii) developing the MenuPlannerHelper App**
<!--##`< Create a DIV to HIDE and SHOW the details >`-->

### -- DATA
- Relative to many online recipes, [BBC Good Food](https://www.bbcgoodfood.com/) has a decent collection of recipes with information on the difficulty of a recipe. 
	
-	I coded a web-recipe scrapper to automatically scrap all available (~10,000 at the time of scraping) recipes from [BBCgoodfood.com](https://www.bbcgoodfood.com/). This included information on Ingredients, Instructions, as well as additional recipe information e.g. difficulty, preparation time, etc.

### -- PRE-PROCESSING
- Like numerical data, text data also requires some form of “pre-processing”, which aims to clean up information that is not task-relevant and/or to restructure the data for subsequent analyses. To this end, I employed techniques from Natural Language Processing (NLP), which can be appreciated as the union of Artificial Intelligence (AI) and linguistics. NLP involves developing and using algorithmic and/or probablistic analysis of written language to automatically derive some insights from text data. 

-	In particular, I borrowed the [Conditional-Random-Field Ingredient Phrase Tagger developed by the New York Times (NYT)](https://open.blogs.nytimes.com/2015/04/09/extracting-structured-data-from-recipes-using-conditional-random-fields/) with optimization and inference using the [CRF++ implementation](https://taku910.github.io/crfpp/) to help with predicting the sequence of ingredient information. I modified the NYT Ingredient Phrase Tagger specifically to help remove quantity and units’ information from each BBCgooodfood recipe’s ingredient phrases and also to retrieve the name(s) of ingredients used. <!--(*NB – the outcome of the modified NYT ingredient phrase tagger also inherits the ~89% phrase-tagging accuracy and while ‘imperfect’ it does a sufficiently decent job!*)-->

-	Among various tweaks to the publicly available [NYT ingredient phrase tagger code](https://github.com/NYTimes/ingredient-phrase-tagger) given the different data structure between [NYT Cooking](https://cooking.nytimes.com/) vs [BBC Good Food](https://www.bbcgoodfood.com/) recipes, as well as what might be deemed as collective unit terms e.g. ‘clove’, ‘bushel’, ‘pinch’, I also modified the NYT Ingredient Phrase Tagger utility code to account for metric units, since the recipes from [BBC Good Food](https://www.bbcgoodfood.com/) are written in British rather than American English. 

-	Subsequently, the recipe ingredient and instruction text data were [tokenized](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html), i.e. they went through an algorithmic process that breaks down strings of words into its linguistic components e.g. words vs. non-words, parts-of-speech etc. so you could choose to keep only those elements of interest. 

### -- TOPIC-MODELING
-	Next, I performed ***topic-modeling*** --- an un-supervised machine learning approach that discovers the associations between words, topics, and documents<!-- (e.g. in the present case, it attempts to associate the ingredient phrases or instructions for each recipe with a topic)-->. --- using [Latent Dirichlet Allocation (LDA)](http://www.cs.columbia.edu/~blei/papers/Blei2012.pdf)<sup>LDA</sup>.<!-- # – distinct from [Linear Discriminant Analysis](https://en.wikipedia.org/wiki/Linear_discriminant_analysis) which is an algorithm that seeks to find a linear combination of features characterizing or separating two or more classes of objects or events)--> 

-  	The LDA topic-model assumes that a specific probabilistic model generates all the documents. Inherent in this assumption is that all documents share the same set of topics, but each document exhibits a mixture of topics (drawn from a Dirichlet<sup>Dir</sup> prior `Dir_a`), with some being more salient than others. The words associated with each topic is related to a multinomial distribution over the range of vocabulary (drawn from a Dirichlet prior `Dir_b`). 
> ### LDA assumption: generated documents consist of distributions of topics, which are distributions of words. 

-	<!--This process describes a generative model wherein--> This means that for any given observed collection of documents, we are trying to infer the latent variables ***i) the probability of words being used for each topic –-- a word-topic association,*** and ***ii) the probability of each topic appearing in each document –-- a topic—document association*** based on observed variables; the vocabulary itself. The inference process is typically derived through [Gibbs sampling](https://en.wikipedia.org/wiki/Gibbs_sampling), <!--an implementation of Markov Chain Monte-Carlo algorithm, -->or formulated as an optimization problem using [variational inference](https://ermongroup.github.io/cs228-notes/inference/variational/), and tuning the two hyper-parameters `a`) and `b`) which regulate the prior distributions. 

- While other topic-models would likely also yield sensible topic clusters; [LDA] (http://www.cs.columbia.edu/~blei/papers/Blei2012.pdf) was opted because its learned topics are generally more concise and coherent, and it is considered [a strong choice for applications in which a human end-user is envisioned to interact with the learned topics](http://aclweb.org/anthology/D/D12/D12-1087.pdf).

-	I implemented LDA separately on tokenized ingredient and instruction text data across all recipes using Python’s [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation) (although you can also do so with the [Gensim](https://radimrehurek.com/gensim/models/ldamodel.html) library). After some iterative process in assessing the number of latent topics<sup>Ntopics</sup> from the collection of recipes<!--** (heuristics in determining number of topics can be found here## -
	https://github.com/nikita-moor/ldatuning
	https://github.com/WZBSocialScienceCenter/tmtoolkit
	http://ellisp.github.io/blog/2017/01/05/topic-model-cv
https://stats.stackexchange.com/questions/295506/lda-topics-number-determining-the-fit-level-with-current-number-of-topics
)-->, the final LDA model yielded generally meaningful ingredient (`N=100`) and instruction (`N=80`) topics. 

![](https://raw.githubusercontent.com/hengrumay/hengrumay.github.io/master/_posts/MenuPlannerHelper/LDA_ingredTopics.png)<center>FIG3: <i>Examples of the varying distributions of Ingredient topics associated with each recipe</i></center>

![](https://raw.githubusercontent.com/hengrumay/hengrumay.github.io/master/_posts/MenuPlannerHelper/LDA_instructTopics.png)<center>FIG4: <i>An illustrative snapshot of Instruction topics visualized using the [interactive LDAviz tool](https://github.com/bmabey/pyLDAvis)</i></center>

### -- CLASSIFICATION
-   With the LDA ingredient and instructions topics derived, I assessed a few Classification Models that included the probabilistic topic-word association matrices as input features to predict recipe difficulty (‘easy’ vs. ‘more challenging’). The general model takes the form (also shown in FIG2.): 
<!--β_0  + β_1 〖LDA〗_ingredients  + β_2 〖LDA〗_instructions  + β_3 〖Time〗_prep  + β_4 〖Time〗_cook+ β_5 N_ingredients  =〖Difficulty 〗_(0=more_challenging)^(1=easy)--> 
<center>![](https://raw.githubusercontent.com/hengrumay/hengrumay.github.io/master/_posts/MenuPlannerHelper/Classification_Model.png)</center>
<!--<math display="block">
	<msubsup><mi>β</mi> <mi>0</mi> <mi></mi></msubsup>
	<mo>+</mo>
	<msubsup><mi>β</mi> <mi>1</mi> <mi></mi></msubsup>
	<msubsup><mi>LDA</mi> <mi>ingredients</mi> <mi></mi></msubsup>
	<mo>+</mo>
	<msubsup><mi>β</mi> <mi>2</mi> <mi></mi></msubsup>
	<msubsup><mi>LDA</mi> <mi>instructions</mi> <mi></mi></msubsup>
	<mo>+</mo>
	<msubsup><mi>β</mi> <mi>3</mi> <mi></mi></msubsup>
	<msubsup><mi>TIME</mi> <mi>prep</mi> <mi></mi></msubsup>
	<mo>+</mo>
	<msubsup><mi>β</mi> <mi>4</mi> <mi></mi></msubsup>
	<msubsup><mi>TIME</mi> <mi>cook</mi> <mi></mi></msubsup>
	<mo>+</mo>
	<msubsup><mi>β</mi> <mi>5</mi> <mi></mi></msubsup>
	<msubsup><mi>N</mi> <mi>ingredients</mi> <mi></mi></msubsup>
   	<mo>=</mo>
   	<msubsup><mi>Difficulty</mi> <mi>0=more_challenging</mi> <mi>1=easy</mi></msubsup> 
</math> -->

-   Ensemble ([Gradient-boosted & Random Forest](https://discuss.analyticsvidhya.com/t/what-is-the-fundamental-difference-between-randomforest-and-gradient-boosting-algorithms/2341)) [classification](http://www.saedsayad.com/decision_tree.htm) [Trees](https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052) and [Logistic Regression](http://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html) Models with different [regularization e.g. Lasso(L1) & Ridge(L2)](https://www.quora.com/Using-logistic-regression-and-L1-L2-regularization-do-I-have-to-care-about-features-selection) parameters were compared.


### --	TESTING-VALIDATING
-	To address the uneven proportion<sup>sample-issue</sup> of recipes for each difficulty category, the number of easy recipes was downsampled to match those of the ‘more-challenging’ recipes. <!--*#* (see *** for other ways to deal with uneven data samples) -->
-	To assess the different models, 20% of sample data was held for final testing, and the remaining 80% was further split into 70% for model training and 30% for model development-testing.
-	The outcome metrics of interest here were area under the curve, as well as precision (% of selected items that are relevant) and recall (% of relevant items selected, also commonly known as '*sensitivity*'):   
<center><img src="https://upload.wikimedia.org/wikipedia/commons/2/26/Precisionrecall.svg" height="600px" align="center"> </center><center>FIG5: <i>Precision and Recall, illustrated -- credit: [Wikipedia](https://upload.wikimedia.org/wikipedia/commons/2/26/Precisionrecall.svg)</i> </center>

-	The different models do comparably well after tuning for their respective parameters (e.g. learning rate | number of trees | training features) with K-fold cross-validation. The 2 best performing models: `Logistic_Regression1_lasso` and `gradboostedTrees` yielded comparable recall and precision metrics ~84—86%, as seen in the confusion matrices below. 

![](https://raw.githubusercontent.com/hengrumay/hengrumay.github.io/master/_posts/MenuPlannerHelper/ClassificationOutcomes.png)<center>FIG6: <i>Comparison of classification outcomes on hold-out data</i></center>


<!--##`</ CLOSE DIV to HIDE and SHOW the details >`-->

<br>

### FEATUREs contributing to recipe difficulty 
With our classification models yeilding reasonable [precision and recall](https://upload.wikimedia.org/wikipedia/commons/2/26/Precisionrecall.svg) metrics, we could start to probe into features contributing to a recipe's preparation ease or complexity: 

![](https://raw.githubusercontent.com/hengrumay/hengrumay.github.io/master/_posts/MenuPlannerHelper/Recipe-Difficulty.png)<center>FIG7: <i>Features (normalized) contributing to making a recipe 'easy' or 'more challenging'</i></center>  

<!--Given that we are interested in understanding what might be relevant features contributing to making a recipe easy or challenging, it is worth choosing the model that provides a more intuitive understanding of how recipes are classified. 
-->In addition to providing a probability of the associated topics, the `LogisticRegression` model with lasso regularization provides us some helpful insights into understanding what might be relevant features contributing to making a recipe easy or more challenging:

- **EASY Recipes:** those with instructions or cooking methods that involve making soups or stews and one-pot dishes (baked or otherwise) as well as ingredients that incorporate general baking, chicken, Italian herbs, or related to finger-food <!--are generally ‘easy’-->

- **More-Challenging Recipes:** those that have ingredients related to using chocolate and instructions or methods that involve e.g. making ganache, pastry, parcels; deep-frying, roasting; or preparing custard and especially meringues <!--are considered ‘more-challenging’.--> 

> ### <!--As someone who has attempted trying a range of recipes and cooking methods, these insights seem somewhat reasonable.--> <!--*Intrestingly, these insights seem to support my suspicion that some finesse is required in becoming even an amateur pastry chef!*-->  

  
<br>

### APPLICATIONs 
Now that we have a working recipe-difficulty-tagger (classification model), we could start tagging (existing and new) recipes within the collection that doesn’t yet have a difficulty category. 

Apart from classifying recipes for their difficulty, I was also interested in providing alternative suggestions that could still make use of ingredients similar to those listed in the original recipe. <!--Indeed we could also develop the MenuPlannerHelper App! --> To this end, I further employed the [K-nearest-neighbors (KNN) algorithm](http://www.saedsayad.com/k_nearest_neighbors.htm) on the LDA document-topic association matrix derived for ingredient topics across all recipes. Doing so, we could find `K` other recipes (documents) whose distribution of ingredient-topics are closest to any recipe of choice (as measured by [cosine-similarity](https://en.wikipedia.org/wiki/Cosine_similarity)). 
<!--[Similar implementation is also described in http://ieeexplore.ieee.org/document/8054520/  ||
| https://doi.org/10.1016/j.proeng.2014.03.129]-->

<!--Using the LDA association matrices, KNN algorithm, I built a minimal web app using Flask, twitter bootstrap, and CSS+HTML. 
-->
Below is an early version demo of the [MenuPlannerHelper](https://bit.ly/menuplannerhelper) App!

<!-- <video controls width="800" height="600" align="center">
	<source src="https://raw.githubusercontent.com/hengrumay/hengrumay.github.io/master/_posts/MenuPlannerHelper/MenuHelper_v13.mp4" type="video/mp4">
  <!--<source src="videos/real-estate.mp4" type="video/mp4">-->
  <!--<source src="videos/real-estate.ogv" type="video/ogg">-->
<!-- </video> -->

<!-- <center>VIDEO DEMO: *The [MenuHelper](https://bit.ly/menuplannerhelper) app is built with [Flask](http://flask.pocoo.org/docs/0.12/), [bootstrap](http://getbootstrap.com/), [CSS](https://developer.mozilla.org/en-US/docs/Web/CSS) + [HTML](https://developer.mozilla.org/en-US/docs/Web/HTML) and hosted on [AWS](https://aws.amazon.com/)*</center> -->

[![](https://raw.githubusercontent.com/hengrumay/hengrumay.github.io/master/_posts/MenuPlannerHelper/MenuPlannerHelper_AppDemo.png)](https://bit.ly/menuplannerhelper)<center>FIG8: <i>The [MenuHelper](https://bit.ly/menuplannerhelper) app is built with [Flask](http://flask.pocoo.org/docs/0.12/), [bootstrap](http://getbootstrap.com/), [CSS](https://developer.mozilla.org/en-US/docs/Web/CSS) + [HTML](https://developer.mozilla.org/en-US/docs/Web/HTML)</i> 
< TRY to embed demo video > 
</center>  

<br>

Happily, the ***Recipe-Difficulty-Tagger*** and ***MenuPlannerHelper App*** serve as decent working proof-of-concepts and can indeed be used with the current recipe collection from [BBCgoodfood.com](https://www.bbcgoodfood.com/). 

<br> 

> ### Some thoughts on how to further improve and extend the work here... 


### APPLICATION EXTENSIONS:

* One could apply the <!--same techniques-->recipe-difficulty-tagger to other (online and/or analog) recipe collection. <!--e.g. NYT Cooking etc.-->  
  
* We could also assess if similar ingredient and instruction topic features overlap across different recipe collections for difficulty. There may be a ‘universal’ set of features that could be approximated (and updated) to tag recipes for difficulty.  

* Since the concept of ‘difficulty’ may be subjective, one could potentially personalize the MenuHelper app to track your culinary adventures over time. This in turn might mean that personal perspectives on what is subjectively easy or challenging could fine-tune the recipe-difficulty-tagger model to fit your level of comfort and may be used to suggest more challenging alternatives should you feel up for it. 

### ROOM for IMPROVEMENTs in data and modeling pipelines:

*	The [issue of unbalanced class proportion](https://svds.com/learning-imbalanced-classes/)<sup>sample-issue</sup> and modeling could be further assessed with [stratified k-fold cross-validation](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) on the training data, adapting the data resampling with bootstrap aggregating (‘bagging’), or indeed adjusting the class-weights.

* Using a more objective way [to select the number of topics in the topic-modeling](http://ellisp.github.io/blog/2017/01/05/topic-model-cv)<sup>Ntopics</sup> , as well as other ways of defining relevant recipe features from recipe instructions e.g. context of words maybe useful and [word2vec](https://arxiv.org/pdf/1310.4546.pdf)<!--(http://adventuresinmachinelearning.com/word2vec-keras-tutorial/)--> may help with finding these.

<br>
I look forward to trying out some of these ideas and sharing updates on another occasion. 

Meanwhile, if you find yourself using recipes from [BBCgoodfood](https://www.bbcgoodfood.com/) why not give the [MenuHelper](bit.ly/menuplannerhelper) App a go! You never know, there could be an easier or similarly delectable recipe that could help simplify your holiday festive meal preparation. 

### *Until next time, Happy Holidays to all!*

<br>
<br>

--
##### *Postcript*: 
*The use of recipes as a motivating dataset for exploring topic-modeling and application development may seem frivolous. However, the methods, learning algorithms, and ideas integrated in the work described here have applications that can be extrapolated to other domains and problems that deal with unstructured text information or others requiring feature engineering.*



--
<br>
====  
##### _FOOTNOTES:_ 

**<sup>LDA:</sup>** An unfortunate sharing of acronym: Latent Dirichlet Allocation (LDA) is distinct from [Linear Discriminant Analysis (LDA)] (https://en.wikipedia.org/wiki/Linear_discriminant_analysis), an algorithm that seeks to find a linear combination of features characterizing or separating two or more classes of objects or events.

**<sup>Dir:</sup>** The "Dirichlet" distribution describes a distribution of distributions.

**<sup>Ntopics:</sup>** A more objective way to determine the 'optimal' number of topics for a corpus of documents is through cross-validating a model's [perplexity](https://en.wikipedia.org/wiki/Perplexity) -- *the measure of how well a probability model predicts a data sample*. An overview of [applying this heuristic](https://doi.org/10.1186/1471-2105-16-S13-S8) using the [ldatuning R package](https://cran.r-project.org/web/packages/ldatuning/index.html) is given [here](http://ellisp.github.io/blog/2017/01/05/topic-model-cv), and a tutorial using the [Text Mining and Topic Modeling Toolkit for Python](https://github.com/WZBSocialScienceCenter/tmtoolkit) can be found [here](https://datascience.blog.wzb.eu/2017/11/09/topic-modeling-evaluation-in-python-with-tmtoolkit/)  

**<sup>sample-issue:</sup>** Some heuristics: [on how to deal with imbalanced data](https://svds.com/learning-imbalanced-classes/) and [on the 'right' way to oversample](https://beckernick.github.io/oversampling-modeling/) for predictive modeling.

**<sup>GITHUB_repo:</sup>** [https://github.com/hengrumay/recipes](https://github.com/hengrumay/recipes) 

<!--^§ The topic and classification modeling as well as the prototype web application detailed here were initially developed during the last 3 weeks of the [METIS data science bootcamp](https://www.thisismetis.com/data-science-bootcamps) in Dec. 2016. Subsequent revisions to the original LDA modeling and assessments were performed to improve interpretation of the model(s). 
-->
