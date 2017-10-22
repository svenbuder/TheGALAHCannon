TheGALAHCannon
==============

This is the code associated with *The Cannon*, a data-driven method for
determining stellar labels (physical parameters and chemical abundances) from
stellar spectra in the context of large spectroscopic surveys. It closely
follows the method outlined in Ness et al. 2015, although some minor changes
have been made. A description of these changes will be outlined in Buder et al. (in prep).

Authors
-------

* **Sven Buder** (MPIA)
* **Melissa Ness** (MPIA)

Files (in order of suggested use)
---------------------------------

1) Notebook to create training set \
   **Cannon_maketraining.ipynb** \
   INPUT: version_cannon, version_reduction, mode

2) Notebook to train and later test based on the training set \
   **Cannon_train.ipynb**
   INPUT: version_cannon, version_reduction, mode, obs_date \

3) Notebook to create test set \
   **Cannon_maketest.ipynb** \
   INPUT: version_reduction

4) Notebook to create abundance training sets (after stellar parameter run) \
   **Cannon_ab_trainingset_preparations.ipynb** \
   INPUT: version_cannon, version_reduction

5) Notebook to collect trainingset and check performance (incl. error estimation) \
   **Cannon_collect_trainingset.ipynb** \
   INPUT: version_cannon, version_reduction

6) Notebook to collect testset of given obs_date \
   **Cannon_collect_test.ipynb** \
   INPUT: version_cannon, version_reduction, obs_date

7) Notebook to stack training set fits and obs_date fits to final sobject_cannon*.fits \
   **Cannon_stack_fits.ipynb** \
   INPUT: version_cannon, version_reduction
