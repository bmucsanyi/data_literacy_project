# Analysing Influential Factors of a Successful Movie

### Abstract

We use the official IMDb datasets extended with scraped content from the IMDb website to predict the IMDb rating based on features such as the length, budget, or genre of a movie. We use logistic regression for this purpose. Furthermore, we monitor the effect of COVID-19 on movie revenues in the form of a hypothesis test.

### Downloading Data
* [Link to the data used](https://drive.google.com/drive/folders/1YYIwQUfeCLxBscTO9i1_9-AoPWsOQbS-?usp=sharing)
* The ``dat`` folder should be copied into the root folder of the repository.
* After that, ``src.preprocess.py`` can be ran to do the preprocessing step on the data.

### Usage
* To recreate figures, use ``exp.fig_1.py``, ``exp.fig_2.py``, ``exp.fig_A1.py``, and ``exp.fig_A2.py``.
* The model evaluation can also be done separately by running ``src.eval_models.py``.
