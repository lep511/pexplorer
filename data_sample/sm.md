# Data Sample

From: https://www.kaggle.com/residentmario/data-types-and-missing-values/tutorial

## **Kepler Exoplanet Search Results (*cumulative.csv*)**

10000 exoplanet candidates examined by the Kepler Space Observatory
Last Updated: 4 years ago (Version 2)

### About this Dataset
Context
The Kepler Space Observatory is a NASA-build satellite that was launched in 2009. The telescope is dedicated to searching for exoplanets in star systems besides our own, with the ultimate goal of possibly finding other habitable planets besides our own. The original mission ended in 2013 due to mechanical failures, but the telescope has nevertheless been functional since 2014 on a "K2" extended mission.

Kepler had verified 1284 new exoplanets as of May 2016. As of October 2017 there are over 3000 confirmed exoplanets total (using all detection methods, including ground-based ones). The telescope is still active and continues to collect new data on its extended mission.

### Content
This dataset is a cumulative record of all observed Kepler "objects of interest" — basically, all of the approximately 10,000 exoplanet candidates Kepler has taken observations on.

This dataset has an extensive data dictionary, which can be accessed here. Highlightable columns of note are:

* kepoi_name: A KOI is a target identified by the Kepler Project that displays at least one transit-like sequence within Kepler time-series photometry that appears to be of astrophysical origin and initially consistent with a planetary transit hypothesis
* kepler_name: [These names] are intended to clearly indicate a class of objects that have been confirmed or validated as planets—a step up from the planet candidate designation.
* koi_disposition: The disposition in the literature towards this exoplanet candidate. One of CANDIDATE, FALSE POSITIVE, NOT DISPOSITIONED or CONFIRMED.
* koi_pdisposition: The disposition Kepler data analysis has towards this exoplanet candidate. One of FALSE POSITIVE, NOT DISPOSITIONED, and CANDIDATE.
* koi_score: A value between 0 and 1 that indicates the confidence in the KOI disposition. For CANDIDATEs, a higher value indicates more confidence in its disposition, while for FALSE POSITIVEs, a higher value indicates less confidence in that disposition.

### Acknowledgements
This dataset was published as-is by NASA. You can access the original table here. More data from the Kepler mission is available from the same source here.

### Inspiration
How often are exoplanets confirmed in the existing literature disconfirmed by measurements from Kepler? How about the other way round?
What general characteristics about exoplanets (that we can find) can you derive from this dataset?
What exoplanets get assigned names in the literature? What is the distribution of confidence scores?

-------

## **Ramen Ratings (*ramen-ratings.csv*)**
Over 2500 ramen ratings
Last Updated: 4 years ago (Version 1)

### About this Dataset
The Ramen Rater is a product review website for the hardcore ramen enthusiast (or "ramenphile"), with over 2500 reviews to date. This dataset is an export of "The Big List" (of reviews), converted to a CSV format.

### Content
Each record in the dataset is a single ramen product review. Review numbers are contiguous: more recently reviewed ramen varieties have higher numbers. Brand, Variety (the product name), Country, and Style (Cup? Bowl? Tray?) are pretty self-explanatory. Stars indicate the ramen quality, as assessed by the reviewer, on a 5-point scale; this is the most important column in the dataset!

Note that this dataset does not include the text of the reviews themselves. For that, you should browse through https://www.theramenrater.com/ instead!

### Acknowledgements
This dataset is republished as-is from the original BIG LIST on https://www.theramenrater.com/.

### Inspiration
What ingredients or flavors are most commonly advertised on ramen package labels?
How do ramen ratings compare against ratings for other food products (like, say, wine)?
How is ramen manufacturing internationally distributed?

-------

## **Wine Reviews (*winemag-data-130k-v2.csv*)**
130k wine reviews with variety, location, winery, price, and description
Last Updated: 4 years ago (Version 4)

### About this Dataset
After watching Somm (a documentary on master sommeliers) I wondered how I could create a predictive model to identify wines through blind tasting like a master sommelier would. The first step in this journey was gathering some data to train a model. I plan to use deep learning to predict the wine variety using words in the description/review. The model still won't be able to taste the wine, but theoretically it could identify the wine based on a description that a sommelier could give. If anyone has any ideas on how to accomplish this, please post them!

### Content
This dataset contains 10 columns and 130k rows of wine reviews.

### Acknowledgements
The data was scraped from WineEnthusiast during the week of June 15th, 2017. The code for the scraper can be found here if you have any more specific questions about data collection that I didn't address.

### Inspiration
I think that this dataset offers some great opportunities for sentiment analysis and other text related predictive models. My overall goal is to create a model that can identify the variety, winery, and location of a wine based on a description. If anyone has any ideas, breakthroughs, or other interesting insights/models please post them.
