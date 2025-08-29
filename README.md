# COVID-19 Assignment

Machine learning pipeline for COVID-19 forecasting.

This project includes:
- **Data loading & preprocessing**
- **Model training** 
- **Prediction generation**
- **Visualization of forecasts vs actuals**

---

## 1. Environment Setup

### Step 1 — Install and use Poetry
Poetry is used for dependency and environment management.  
Follow the [Poetry installation guide](https://python-poetry.org/docs/#installation).


If you are modifying the toml file. Follow these steps :
```
poetry lock OR poetry lock --no-update
potery install
```
If you add the library like poetry add torchvision@0.17.2+cpu, it will automatically lock the file with poetry lock --no-update


Removing/Adding packages to Poetry Manually . Follow these steps :

```
poetry remove pandas


poetry add pandas --source pytorch
```


Validate if Package is installed. Follow these steps :
```
poetry run python -c "import pandas; print(pandas.version)"
```





### Step 2.1 - If there was a repo it would be (no repo as I didn't want to share ongoing recrutation solution)
```bash
git clone <repo-url>
cd <repo-root>
```

### Step 2.2 - Instead of cloning the repo, unzip the attachemnt from mail and move to the directory where you saved project:
```
cd <location-of-unzipping>\Project_ML\Assignment>
```

### Step 3 - Install dependencies 

```
poetry install
```

## 2. Running The Pipeline

### Train models
Uses the latest trained models to predict and saves results to:
```
predictions\predictions_<timestamp>.csv
```
Run
```
poetry run train
```

### Generate predictions
Uses the most recent trained models:
```
poetry run predict
```

### Plot results
Loads the most recent predictions CSV and generates comparison plots:

```
poetry run plot
```

## 3. Formatting
This project uses black for consistent code formatting:
```
poetry run black .
```
## 4. Project Commands Overview
| Command                      | Description                                                 |
| ---------------------------- | ----------------------------------------------------------- |
| `poetry run train`           | Train models on the dataset (skips if models already exist) |
| `poetry run predict`         | Generate predictions from latest trained models             |       |
| `poetry run plot`            | Plot results from the latest predictions file               |
| `poetry run black .`         | Format all code with Black                                  |
## 5. Requirements

- All the requirements are listed in pyproject.toml file

## 6. Author thoughts/comments
- There is a few things that I would do if I had more time and more context provided:
    - I would change assignment folder name to src (didn't catch it soon enough to change the paths from imports etc.)
    - Create tests and the whole CI/CD pipeline for it as for any version control action it would be nice to have.
    - Adjust the project to the user demands (do we want to just use it for continous training or do we want this whole thing to just predict? I was not sure which one is it so i did something in between)
    - I would create even more services (e.g. in features there is a part that could be a separate script like location transformations)
    - Create more automated data downlad and data choose options, so far the only way to change data is to change paths to it's download.
    - Because the data is not changing and covid-19 thing is quite a history I didn't interfere into last training/eval datestamps but If we were handling something like timeseries BTC/USD prediciton with live yahoo data I would actually make it a bit more dynamic.
    - Some additional linting could be added but it's more like CI/CD pipeline to be created 
