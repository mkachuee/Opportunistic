
### For a detailed explanation of the methods used here for the cost-sensetive health dataset, please refer to: ["Nutrition and Health Data for Cost-Sensitive Learning"](https://github.com/mkachuee/OpportunisticData/raw/master/Paper.pdf)

### For a detailed explanation of the opportunsic learning method, please refer to: ["Opportunistic Learning: Budgeted Cost-Sensitive Learning from Data Streams"](https://openreview.net/forum?id=S1eOHo09KX)

## File list:
- **nhanes.py:** implementation of the data preprocessing logic as well as definition of a few example datasets such as diabetes, heart disease, hypertention, etc.
- **Demo_Dataset.ipynb:** Jupyter notebook file to demonstrate the basic usage of each sample dataset.
- **Demo_OL_DQN.ipynb:** Jupyter notebook file to demonstrate a simple implementation of the Opportunistic Learning method.
- Other source files are used in the Demo_OL_DQN.ipynb.


## How to use:
1) Download [raw data files](https://drive.google.com/file/d/1hFp7O747408D8t5442f0Sjit7wXKXI1z/view?usp=sharing) and decompress them.
2) Install Python 3 and the following packages: joblib, numpy, pandas, matplotlib, scipy, sklearn, jupyter, pytorch.
3) Use Demo_Dataset.ipynb and Demo_OL_DQN.ipynb to see a few examples on how to use the predefined tasks.
4) Alternatively, you can expand nhanes.py to define new tasks by following the implementation logic of the provided samples.

## Citation Request
If you find this repository useful, please cite the following papers:

* M. Kachuee, O. Goldstein, K. Kärkkäinen, S. Darabi, M. Sarrafzadeh, Opportunistic Learning: Budgeted Cost-Sensitive Learning from Data Streams, International Conference on Learning Representations (ICLR), 2019. [Paper](https://openreview.net/forum?id=S1eOHo09KX)
* M. Kachuee, K. Kärkkäinen, O. Goldstein, D. Zamanzadeh, M. Sarrafzadeh, Nutrition and Health Data for Cost-Sensitive Learning, 2019. 
