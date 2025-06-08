# Predicting sedimentation velocity using ML  
ML project for RFP EDF aiming to calculate sedimentation rate.
## Project structure
```markdown
project-root/
├── data/
│   ├── raw/     # contains raw .csv file
│   ├── interim/ # contains .csv with no duplicates and NaNs replaced by mean value
│   └── processed/
├── notebooks/
│   ├── EDA_ml_sediment.ipynb     # exploratory data analysis with visuals
│   ├── modular_ml_sediment.ipynb # modular code
├── src/
│   ├── data_utils.py     # data loading & cleaning
│   ├── preprocessing.py  # preproc pipelines
│   └── models.py         # create and train models
├── tests/
├── models/
├── reports/
├── config/
└── experiments/
```
## Next