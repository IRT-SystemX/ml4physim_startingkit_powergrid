# Starting Kit - Machine Learning for Physical Simulation Challenge
This starting kit provides a set of jupyter notebooks helping the challenge participants to better understand the use case, the dataset and how to contribute to this competition. For general information concerning the challenge and submit your solutions, you can refer to the competition [Codabench page](https://www.codabench.org/competitions/2378/).

Data
----
A demo version of the usecase datasets is provided in `input_data_local` for 3 different scales of Power Grid environments. Two environments `lips_case14_sandbox` and `lips_neurips_2020_track1_small` includes 14 and 38 nodes respectively and provided for participants to test (optional) their solution. Environment `lips_idf_2023` is the challenge environment. All the final solutions (which are submitted on codabench) should be trained and evaluated on this environment. See the description of provided jupyter notebooks below to see how to use and import these datasets.

A `configs` folder is provided which includes the configurations (parameters) related to benchmarks and augmented simulators (aka models). The users could change the simulators configurations to change the hyperparameter of existing models. More details on how to use these configs is provided in the notebooks.

Finally, the `trained_models` inside `input_data_local` directory contains a set of trained baseline models, which are used to show the evaluation procedure and scoring.

Prerequisites
--------------
Most of the notebooks provided in this repository are based on LIPS platform. To be able to execute the jupyter notebooks provided in this repository and described in the following section, the [LIPS platform](https://lips.irt-systemx.fr/) should be installed properly. The installation procedure is explained in the [LIPS package repository](https://github.com/IRT-SystemX/LIPS), in [this section](https://github.com/IRT-SystemX/LIPS#installation).

To get familiar with LIPS platform itself, it includes its own set of [jupyter notebooks](https://github.com/IRT-SystemX/LIPS/tree/main/getting_started). For this competition, the participants may focus on [these notebooks](https://github.com/IRT-SystemX/LIPS/tree/main/getting_started/PowerGridUsecase) (focusing on Power Grid use case) provided in LIPS package.  

Notebooks description
---------------------
In the following, we describe the content of the jupyter notebooks : 

- **0_Basic_Competition_Information**: This notebook contains general information concerning the competition organization, phases, deadlines and terms. The content is the same as the one shared in the competition Codabench page. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/IRT-SystemX/ml4physim_startingkit_powergrid/blob/main/0_Basic_Competition_Information.ipynb) 

- **1-PowerGrid_basics**: This notebook aims to familiarize the participants with the use case and to facilitate their comprehension. It allows the visualization of some simulation results. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IRT-SystemX/ml4physim_startingkit_powergrid/blob/main/1_PowerGrid_Usecase_basics.ipynb)

- **2-Datasets**: Shows how the challenge datasets could be downloaded and imported using proper functions. These data will be used in the following notebook to train and evaluate an augmented simulator. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IRT-SystemX/ml4physim_startingkit_powergrid/blob/main/2-Datasets.ipynb) 

- **3-Reproduce_baseline**: This notebook shows how the baseline results could be reproduced. It includes the whole pipeline of training, evaluation and score calculation of an augmented simulator using [LIPS platform](https://github.com/IRT-SystemX/LIPS). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IRT-SystemX/ml4physim_startingkit_powergrid/blob/main/3_Reproduce_baseline.ipynb) 

- **4a-How_to_Contribute_Pytorch**: This notebook shows 3 ways of contribution for beginner, intermediate and advanced users. The submissions should respect one of these forms to be valid and also to enable their proper evaluation through the LIPS platform which will be used for the final evaluation of the results. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IRT-SystemX/ml4physim_startingkit_powergrid/blob/main/4a_How_to_Contribute_Pytorch.ipynb)

       * Beginner Contributor: You only have to calibrate the parameters of existing augmented simulators
       * Intermediate Contributor: You can implement an augmented simulator respecting a given template (provided by the LIPS platform)
       * Advanced Contributor: you can implement your architecture independently from LIPS platform and use only the evaluation part of the framework to assess your model performance.

- **4b-How_to_Contribute_Tensorflow**: This notebook shows how to contribute using the existing augmen+ted simulators based on Tensorflow library. The procedure to customize the architecture is fairly the same as pytorch (shown in Notebook 4). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IRT-SystemX/ml4physim_startingkit_powergrid/blob/main/4b_How_to_Contribute_Tensorflow.ipynb)

- **5-Scoring**: This notebook shows firstly how the score is computed by describing its different components. Next, it provides a script which can be used locally by the participants to obtain a score for their contributions. We encourage participants to evaluate their solutions via codabench (which uses the same scoring module as the one described in this notebook). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IRT-SystemX/ml4physim_startingkit_powergrid/blob/main/5_Scoring.ipynb)

- **6-Submission:** This notebook presents the composition of a submission bundle for [Codabench](https://www.codabench.org/competitions/2378/) and usable parameters. 

- **7-Submission_examples:** This notebook shows how to submit on [Codabench](https://www.codabench.org/competitions/2378/) and examples of submissions bundles.  
