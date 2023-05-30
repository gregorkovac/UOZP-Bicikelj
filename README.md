# UOZP - Bicikelj

This is a repository for the project assignment **Bicikelj** at the course *Introduction to Data Mining* at the *Faculty of Computer Science, University of Ljubljana.*

We were given a dataset of the following structure:

| Timestamp      | Station1 | Station2 | ... |
| ----------- | ----------- | ----------- | ----------- |
| `timestamp1`      | `number of bikes`      | `number of bikes`      | ...       |
| `timestamp2`      | `number of bikes`      | `number of bikes`      | ...       |
| ...     | ...     | ...    | ...       |

We had to use machine learning techniques to predict the number of bikes at the stations at some time. The timestamps to be predicted were taken from the training data set. Some of them have a one hour break before and some of them two.

## Project structure
- [bicikelj_train.csv](./bicikelj_train.csv) - the training dataset
- [bicikelj_test.csv](./bicikelj_test.csv) - the dataset to be predicted
- [bicikelj_out_25_05-18-26-58.csv](./bicikelj_out_25_05-18-26-58.csv) - the latest predictions
- [predict.ipynb](./predict.ipynb) - the full analysis and prediction process
- [Poročilo.pdf](./Poro%C4%8Dilo.pdf) - the project report (in Slovenian)