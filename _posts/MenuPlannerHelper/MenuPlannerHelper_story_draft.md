**Once again, the holiday season is upon us.** As you go about your celebrations, you may be both excited and overwhelmed by all the various gatherings and parties. Some of us will attend as guests, and perhaps a few of us might be hosting. Should you find yourself preparing the whole meal or offering to contribute a dish or two and in the mood for homemade culinary adventures, there’s a little web application, called the *[MenuPlannerHelper](https://github.com/hengrumay/recipes)* (abbreviated as ***[MenuHelper](https://bit.ly/menuplannerhelper)***) I developed a while back that could come in handy.

*The inspiration for its creation come from the fact that while I generally browse for new recipes to try, I sometimes find myself in situations where there will be recipes with accompanying food photography that look amazing and seemingly ‘doable’, yet upon embarking on the preparations, they present as more challenging than originally anticipated. In those scenarios, I find myself wishing that there might be an opportunity to use the ingredients for another similar but ‘simpler’ recipe. Likewise, there are other occasions, when a recipe turns out somewhat less complicated and you might wish to add some sophistication without over-tweaking it. Having cooked with others and listened to various kitchen nightmares, I have reason to believe that my experiences are not unique.*

When festivities and inherent traditions place us under time, logistical and resource constraints, you don’t necessarily have much room to “mess up” when you are preparing a meal for others. With various things to coordinate, the last thing we really need while multi-tasking is a kitchen disaster! 
> **It was developed to help categorize recipes into different levels of complexity so that one could experience less stress and more enjoyment in coordinating the preparation of various dishes for e.g. a dinner party. Importantly, because of how it is trained to categorize recipes for relative preparation difficulty, one could further associate recipes based on the degree of similarity between their ingredients. This becomes handy if you wish to pivot between difficulty levels but also use similar ingredients that you may already have available.**
**Not all recipes are created equal** – I learnt this the hard way. Recipes differ on the number of ingredients whether they are exotic or locally sourced, how ingredients are prepared, the type of cooking technique or the equipment required, and so forth. If there would be a way to classify recipes as ‘easy’ or ‘challenging’ perhaps this information could help one to better plan and prepare the combination of dishes for a festive meal. 

![](file:///Users/hrm/Dropbox/RecentProjects/DataStories_github_io/MenuPlannerHelper/NotAllRecipesCreatedEqual.png)<center>FIG1: *The many ways that make each recipe different*</center>

Interestingly, as I sifted through various sources of recipes the information about its preparation difficulty is not always available, nor is it often explicitly stated. It is possible that how difficult a recipe is may depend on your cooking experience, and it might be best left to self-discovery. Yet basic and advanced cooking classes exist, so it might be worth asking:  
![](file:///Users/hrm/Dropbox/RecentProjects/DataStories_github_io/MenuPlannerHelper/CreateNUseRecipeDiffTagger.png)<center>FIG2: *A visual summary of the steps taken to create and use a Recipe-Difficulty-Tagger*</center>
<br>



### Details of **i) building a Recipe-Difficulty-Tagger** and **ii) developing the MenuPlannerHelper App**
<!--##`< Create a DIV to HIDE and SHOW the details >`-->

### -- DATA
- Relative to many online recipes, [BBC Good Food](https://www.bbcgoodfood.com/) has a decent collection of recipes with information on the difficulty of a recipe. 

> ###LDA assumption: generated documents consist of distributions of topics, which are distributions of words. 

-	<!--This process describes a generative model wherein--> This means that for any given observed collection of documents, we are trying to infer the latent variables ***i) the probability of words being used for each topic –-- a word-topic association,*** and ***ii) the probability of each topic appearing in each document –-- a topic—document association*** based on observed variables; the vocabulary itself. The inference process is typically derived through [Gibbs sampling](https://en.wikipedia.org/wiki/Gibbs_sampling), <!--an implementation of Markov Chain Monte-Carlo algorithm, -->or formulated as an optimization problem using [variational inference](https://ermongroup.github.io/cs228-notes/inference/variational/), and tuning the two hyper-parameters \\(a\\) and \\(b\\) which regulate the prior distributions. 

- While other topic-models would likely also yield sensible topic clusters; [LDA] (http://www.cs.columbia.edu/~blei/papers/Blei2012.pdf) was opted because its learned topics are generally more concise and coherent, and it is considered [a strong choice for applications in which a human end-user is envisioned to interact with the learned topics](http://aclweb.org/anthology/D/D12/D12-1087.pdf).

-	I implemented LDA separately on tokenized ingredient and instruction text data across all recipes using Python’s [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation) (although you can also do so with the [Gensim](https://radimrehurek.com/gensim/models/ldamodel.html) library). After some iterative process in assessing the number of latent topics^Ntopics from the collection of recipes<!--** (heuristics in determining number of topics can be found here## -

![](file:///Users/hrm/Dropbox/RecentProjects/DataStories_github_io/MenuPlannerHelper/LDA_instructTopics.png)<center>FIG4: *An illustrative snapshot of Instruction topics visualized using the [interactive LDAviz tool](https://github.com/bmabey/pyLDAvis)*</center>
<math display="block">
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
</math> 
### FEATUREs contributing to recipe difficulty 
With our classification models yeilding reasonable [precision and recall](https://upload.wikimedia.org/wikipedia/commons/2/26/Precisionrecall.svg) metrics, we could start to probe into features contributing to a recipe's preparation ease or complexity: 


-->In addition to providing a probability of the associated topics, the `LogisticRegression` model with lasso regularization provides us some helpful insights into understanding what might be relevant features contributing to making a recipe easy or more challenging:



Apart from classifying recipes for their difficulty, I was also interested in providing alternative suggestions that could still make use of ingredients similar to those listed in the original recipe. <!--Indeed we could also develop the MenuPlannerHelper App! --> To this end, I further employed the [K-nearest-neighbors (KNN) algorithm](http://www.saedsayad.com/k_nearest_neighbors.htm) on the LDA document-topic association matrix derived for ingredient topics across all recipes. Doing so, we could find `K` other recipes (documents) whose distribution of ingredient-topics are closest to any recipe of choice (as measured by [cosine-similarity](https://en.wikipedia.org/wiki/Cosine_similarity)). <!--[Similar implementation is also described in http://ieeexplore.ieee.org/document/8054520/  ||

<!--Using the LDA association matrices, KNN algorithm, I built a minimal web app using Flask, twitter bootstrap, and CSS+HTML. 
-->
Below is an early version demo of the [MenuPlannerHelper](https://bit.ly/menuplannerhelper) App!

<video controls width="800" height="600" align="center">
	<source src="file:///Users/hrm/Dropbox/RecentProjects/DataStories_github_io/MenuPlannerHelper/MenuHelper_v13.mov" type="video/mp4">
  <!--<source src="videos/real-estate.mp4" type="video/mp4">-->
  <!--<source src="videos/real-estate.ogv" type="video/ogg">-->
</video><center>VIDEO DEMO: *The [MenuHelper](https://bit.ly/menuplannerhelper) app is built with [Flask](http://flask.pocoo.org/docs/0.12/), [bootstrap](http://getbootstrap.com/), [CSS](https://developer.mozilla.org/en-US/docs/Web/CSS) + [HTML](https://developer.mozilla.org/en-US/docs/Web/HTML) and hosted on [AWS](https://aws.amazon.com/)*</center> 

<!--[![](file:///Users/hrm/Dropbox/RecentProjects/DataStories_github_io/MenuPlannerHelper/MenuPlannerHelper_AppDemo.png)](https://bit.ly/menuplannerhelper)<center>FIG8: *The [MenuHelper](https://bit.ly/menuplannerhelper) app is built with [Flask](http://flask.pocoo.org/docs/0.12/), [bootstrap](http://getbootstrap.com/), [CSS](https://developer.mozilla.org/en-US/docs/Web/CSS) + [HTML](https://developer.mozilla.org/en-US/docs/Web/HTML)* 
< TRY to embed demo video > 
</center> -->  

<br>


<br> 
> ### Some thoughts on how to further improve and extend the work here... 



  
* We could also assess if similar ingredient and instruction topic features overlap across different recipe collections for difficulty. There may be a ‘universal’ set of features that could be approximated (and updated) to tag recipes for difficulty.  

* Since the concept of ‘difficulty’ may be subjective, one could potentially personalize the MenuHelper app to track your culinary adventures over time. This in turn might mean that personal perspectives on what is subjectively easy or challenging could fine-tune the recipe-difficulty-tagger model to fit your level of comfort and may be used to suggest more challenging alternatives should you feel up for it. 


* Using a more objective way [to select the number of topics in the topic-modeling](http://ellisp.github.io/blog/2017/01/05/topic-model-cv)^Ntopics , as well as other ways of defining relevant recipe features from recipe instructions e.g. context of words maybe useful and [word2vec](https://arxiv.org/pdf/1310.4546.pdf)<!--(http://adventuresinmachinelearning.com/word2vec-keras-tutorial/)--> may help with finding these.

Meanwhile, if you find yourself using recipes from [BBCgoodfood](https://www.bbcgoodfood.com/) why not give the [MenuHelper](bit.ly/menuplannerhelper) App a go! You never know, there could be an easier or similarly delectable recipe that could help simplify your holiday festive meal preparation. 
<br>

--
*The use of recipes as a motivating dataset for exploring topic-modeling and application development may seem frivolous. However, the methods, learning algorithms, and ideas integrated in the work described here have applications that can be extrapolated to other domains and problems that deal with unstructured text information or others requiring feature engineering.*
<br>
##### _FOOTNOTES:_ 

**^(LDA)** An unfortunate sharing of acronym: Latent Dirichlet Allocation (LDA) is distinct from [Linear Discriminant Analysis (LDA)] (https://en.wikipedia.org/wiki/Linear_discriminant_analysis), an algorithm that seeks to find a linear combination of features characterizing or separating two or more classes of objects or events.

**^(Dir)** The "Dirichlet" distribution describes a distribution of distributions.

**^Ntopics** A more objective way to determine the 'optimal' number of topics for a corpus of documents is through cross-validating a model's [perplexity](https://en.wikipedia.org/wiki/Perplexity) -- *the measure of how well a probability model predicts a data sample*. An overview of [applying this heuristic](https://doi.org/10.1186/1471-2105-16-S13-S8) using the [ldatuning R package](https://cran.r-project.org/web/packages/ldatuning/index.html) is given [here](http://ellisp.github.io/blog/2017/01/05/topic-model-cv), and a tutorial using the [Text Mining and Topic Modeling Toolkit for Python](https://github.com/WZBSocialScienceCenter/tmtoolkit) can be found [here](https://datascience.blog.wzb.eu/2017/11/09/topic-modeling-evaluation-in-python-with-tmtoolkit/)  

**^sample-issue** Some heuristics: [on how to deal with imbalanced data](https://svds.com/learning-imbalanced-classes/) and [on the 'right' way to oversample](https://beckernick.github.io/oversampling-modeling/) for predictive modeling.

**^GITHUB_repo** https://github.com/hengrumay/recipes 

<!--^§ The topic and classification modeling as well as the prototype web application detailed here were initially developed during the last 3 weeks of the [METIS data science bootcamp](https://www.thisismetis.com/data-science-bootcamps) in Dec. 2016. Subsequent revisions to the original LDA modeling and assessments were performed to improve interpretation of the model(s). 