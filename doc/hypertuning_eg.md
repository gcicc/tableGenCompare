import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp, chi2_contingency
from ctgan import CTGAN
import logging

# Set up logging
optuna.logging.set_verbosity(optuna.logging.INFO)
logging.basicConfig(level=logging.INFO)

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import LabelEncoder


def evaluate_similarity(original: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    similarity_scores = {}

    for column in original.columns:
        # Skip target variable
        if column == "result_2":
            continue

        # Numerical columns
        if np.issubdtype(original[column].dtype, np.number):
            # Compute Wasserstein Distance
            distance = wasserstein_distance(original[column], synthetic[column])
            # Normalize Wasserstein Distance
            normalized_score = 1 / (1 + distance)
            similarity_scores[column] = normalized_score  # Higher is better

        # Categorical columns
        else:
            # Combine unique categories
            categories = np.union1d(original[column].unique(), synthetic[column].unique())

            # Encode categories and get probabilities
            label_encoder = LabelEncoder()
            encoded_categories = label_encoder.fit(categories)

            orig_probs = (
                original[column].value_counts(normalize=True)
                .reindex(categories, fill_value=0).values
            )
            synth_probs = (
                synthetic[column].value_counts(normalize=True)
                .reindex(categories, fill_value=0).values
            )

            # Compute Jensen-Shannon Divergence
            js_divergence = jensenshannon(orig_probs, synth_probs)
            # Transform JSD to a similarity score
            similarity_score = 1 - js_divergence
            similarity_scores[column] = similarity_score  # Higher is better

    return similarity_scores

from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import ot  # POT library for optimal transport


def bivariate_wasserstein(original: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    """
    Compute a bivariate Wasserstein Distance similarity score for all pairs of continuous columns.
    
    Args:
        original (pd.DataFrame): Original dataset.
        synthetic (pd.DataFrame): Synthetic dataset.

    Returns:
        dict: A dictionary with pairs as keys and normalized Bivariate Wasserstein scores as values.
    """
    similarity_scores = {}
    continuous_cols = original.select_dtypes(include=[np.number]).columns  # Only continuous columns

    # Compute pairwise bivariate Wasserstein Distance for all pairs of continuous variables
    for i, col_x in enumerate(continuous_cols):
        for j, col_y in enumerate(continuous_cols):
            if i >= j:  # Avoid duplicate pairs and self-comparison (e.g., col_x vs col_x)
                continue

            # Create 2D arrays for the original and synthetic datasets
            original_2d = original[[col_x, col_y]].to_numpy()
            synthetic_2d = synthetic[[col_x, col_y]].to_numpy()

            # Compute pairwise Euclidean distances as cost matrix
            cost_matrix = euclidean_distances(original_2d, synthetic_2d)

            # Calculate Wasserstein Distance using the cost matrix
            n, m = cost_matrix.shape
            p_real = np.ones(n) / n  # Uniform empirical distribution for original
            p_synthetic = np.ones(m) / m  # Uniform empirical distribution for synthetic
            wasserstein_dist = ot.emd2(p_real, p_synthetic, cost_matrix)  # 2nd Wasserstein distance

            # Normalize Wasserstein Distance and compute similarity score
            normalized_score = 1 / (1 + wasserstein_dist)  # Normalize to [0, 1]
            similarity_scores[(col_x, col_y)] = normalized_score  # Store result for the variable pair

    return similarity_scores

def evaluate_overall_similarity(original: pd.DataFrame, synthetic: pd.DataFrame) -> tuple:
    """
    Compute overall similarity by combining univariate and bivariate similarity metrics.

    Args:
        original (pd.DataFrame): Original dataset.
        synthetic (pd.DataFrame): Synthetic dataset.

    Returns:
        tuple: Final similarity score, univariate similarity, bivariate similarity
               (higher is better for all scores).
    """
    # 1. Compute univariate similarity
    univariate_similarity = np.mean(list(evaluate_similarity(original, synthetic).values()))

    # 2. Compute bivariate similarity
    bivariate_scores = bivariate_wasserstein(original, synthetic)  # Using continuous columns
    bivariate_similarity = np.mean(list(bivariate_scores.values()))  # Average across pairs

    # 3. Combine metrics using weights
    final_similarity = 0.4 * univariate_similarity + 0.6 * bivariate_similarity

    return final_similarity, univariate_similarity, bivariate_similarity


def objective_with_data(trial, df):
    """
    Evaluate CTGAN hyperparameters using the provided dataset via Optuna search space.

    """
    # Suggest hyperparameters for CTGAN
    epochs = trial.suggest_int("epochs", 500, 5000, step=500)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024])

    # Encode generator and discriminator dimensions as strings
    generator_dim_choices = ['256_256_256', '512_512_512', '128_256_512', '256_128_64']
    discriminator_dim_choices = ['256_256_256', '512_512_512', '1024_1024_1024']
    
    generator_dim = trial.suggest_categorical("generator_dim", generator_dim_choices)
    discriminator_dim = trial.suggest_categorical("discriminator_dim", discriminator_dim_choices)

    # Decode the chosen string into a tuple
    generator_dim = tuple(map(int, generator_dim.split('_')))
    discriminator_dim = tuple(map(int, discriminator_dim.split('_')))

    pac = trial.suggest_categorical("pac", [1, 2, 4, 8, 16])
    generator_lr = trial.suggest_float("generator_lr", 0.00005, 0.0002, log=True)
    discriminator_lr = trial.suggest_float("discriminator_lr", 0.00005, 0.0002, log=True)

    # Remove dropout parameter as it's not supported by CTGAN
    # generator_dropout = trial.suggest_float("generator_dropout", 0.0, 0.5, step=0.1)
    # discriminator_dropout = trial.suggest_float("discriminator_dropout", 0.0, 0.5, step=0.1)

    # Train CTGAN with these hyperparameters
    ctgan = CTGAN(
        generator_dim=generator_dim,
        discriminator_dim=discriminator_dim,
        pac=pac,
        generator_lr=generator_lr,
        discriminator_lr=discriminator_lr,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )

    # Fit CTGAN to the data
    ctgan.fit(df)

    # Generate synthetic samples
    samples = ctgan.sample(len(df))
    df_generated = pd.DataFrame(samples, columns=df.columns)

    # Evaluate overall similarity
    final_similarity, univariate_similarity, bivariate_similarity = evaluate_overall_similarity(df, df_generated)

    # Log similarities for Optuna's trial
    trial.set_user_attr("univariate_similarity", univariate_similarity)
    trial.set_user_attr("bivariate_similarity", bivariate_similarity)
    trial.set_user_attr("similarity", final_similarity)

    # Evaluate downstream task performance
    # Split df into training and testing sets
    X_real = df.drop(columns=["result_2"])
    y_real = df["result_2"]
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42
    )

    # Split df_generated into training and testing sets
    X_synthetic = df_generated.drop(columns=["result_2"])
    y_synthetic = df_generated["result_2"]
    X_synthetic_train, X_synthetic_test, y_synthetic_train, y_synthetic_test = train_test_split(
        X_synthetic, y_synthetic, test_size=0.2, random_state=42
    )

    # Initialize Random Forest Classifier
    clf = RandomForestClassifier(random_state=42)

    # a) Train on synthetic training and test on real testing
    clf.fit(X_synthetic_train, y_synthetic_train)
    acc_a = accuracy_score(y_real_test, clf.predict(X_real_test))

    # b) Train on synthetic training and test on synthetic testing
    clf.fit(X_synthetic_train, y_synthetic_train)
    acc_b = accuracy_score(y_synthetic_test, clf.predict(X_synthetic_test))

    # c) Train on real training and test on real testing
    clf.fit(X_real_train, y_real_train)
    acc_c = accuracy_score(y_real_test, clf.predict(X_real_test))

    # d) Train on real training and test on synthetic testing
    clf.fit(X_real_train, y_real_train)
    acc_d = accuracy_score(y_synthetic_test, clf.predict(X_synthetic_test))

    # Log accuracies
    trial.set_user_attr("acc_a", acc_a)  # Synthetic -> Real
    trial.set_user_attr("acc_b", acc_b)  # Synthetic -> Synthetic
    trial.set_user_attr("acc_c", acc_c)  # Real -> Real
    trial.set_user_attr("acc_d", acc_d)  # Real -> Synthetic

    # Calculate the overall average accuracy
    avg_accuracy = np.mean([acc_a, acc_b, acc_c, acc_d])

    # Combine scores
    combined_score = (0.6 * final_similarity) + (0.4 * avg_accuracy)

    # Print trial progress
    print(
        f"Trial {trial.number}: Univariate Similarity = {univariate_similarity:.4f}, "
        f"Bivariate Similarity = {bivariate_similarity:.4f}, "
        f"Avg Similarity = {final_similarity:.4f}, "
        f"Accuracy = {avg_accuracy:.4f}, Combined Score = {combined_score:.4f}"
    )

    return combined_score


def run_study(df_input, n_trials=25, logfile="study_results.csv"):
    """
    Run an Optuna study with a given DataFrame and specified number of trials.

    Args:
        df_input (pd.DataFrame): Dataset to evaluate.
        n_trials (int): Number of trials for Optuna.
        logfile (str): Path to save trial results to a CSV file.
    """
    global df_train  # Defines the global DataFrame that is passed into the objective function
    df_train = df_input

    def logging_callback(study, trial):
        """
        A callback to log hyperparameters, metrics, and results to a CSV file after each trial.
        """
        trial_data = {
            "trial_number": trial.number,
            "value": trial.value,  # Combined score
            "acc_a_synthetic_real": trial.user_attrs.get("acc_a"),
            "acc_b_synthetic_synthetic": trial.user_attrs.get("acc_b"),
            "acc_c_real_real": trial.user_attrs.get("acc_c"),
            "acc_d_real_synthetic": trial.user_attrs.get("acc_d"),
            "avg_similarity": trial.user_attrs.get("similarity"),
            "univariate_similarity": trial.user_attrs.get("univariate_similarity"),
            "bivariate_similarity": trial.user_attrs.get("bivariate_similarity"),
            "epochs": trial.params.get("epochs"),
            "batch_size": trial.params.get("batch_size"),
            "generator_dim": trial.params.get("generator_dim"),
            "discriminator_dim": trial.params.get("discriminator_dim"),
            "pac": trial.params.get("pac"),
            "generator_lr": trial.params.get("generator_lr"),
            "discriminator_lr": trial.params.get("discriminator_lr"),
        }

        # Decode string-based tuples back to lists for logging clarity
        trial_data["generator_dim"] = tuple(map(int, trial_data["generator_dim"].split('_')))
        trial_data["discriminator_dim"] = tuple(map(int, trial_data["discriminator_dim"].split('_')))

        # Check if the CSV file exists
        try:
            results_df = pd.read_csv(logfile)
        except FileNotFoundError:
            results_df = pd.DataFrame(columns=trial_data.keys())  # Create a new DataFrame if it doesn't exist

        # Add the trial results
        trial_row = pd.DataFrame([trial_data])
        results_df = pd.concat([results_df, trial_row], ignore_index=True)

        # Save the DataFrame to a CSV file
        results_df.to_csv(logfile, index=False)
        print(f"Logged trial {trial.number} results to {logfile}")

    # Create and optimize the Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, callbacks=[logging_callback])

    # Print the best trial results after all trials complete
    print("\nBest hyperparameters:")
    print(study.best_trial.params)
    print("\nBest combined score:")
    print(study.best_value)
    """
    Logs hyperparameters, individual accuracies, and metrics to a CSV file after each iteration.
    """
    trial_data = {
        "trial_number": trial.number,
        "value": trial.value,
        "acc_a_synthetic_real": trial.user_attrs.get("acc_a"),
        "acc_b_synthetic_synthetic": trial.user_attrs.get("acc_b"),
        "acc_c_real_real": trial.user_attrs.get("acc_c"),
        "acc_d_real_synthetic": trial.user_attrs.get("acc_d"),
        "avg_similarity": trial.user_attrs.get("similarity"),
        "univariate_similarity": trial.user_attrs.get("univariate_similarity"),
        "bivariate_similarity": trial.user_attrs.get("bivariate_similarity"),
        "epochs": trial.params.get("epochs"),
        "batch_size": trial.params.get("batch_size"),
        "generator_dim": trial.params.get("generator_dim"),
        "discriminator_dim": trial.params.get("discriminator_dim"),
        "pac": trial.params.get("pac"),
        "generator_lr": trial.params.get("generator_lr"),
        "discriminator_lr": trial.params.get("discriminator_lr"),
    }

    # Convert string-encoded tuples back to readable format in the log
    trial_data["generator_dim"] = tuple(map(int, trial_data["generator_dim"].split('_')))
    trial_data["discriminator_dim"] = tuple(map(int, trial_data["discriminator_dim"].split('_')))

    # Load existing data or create a new DataFrame
    try:
        results_df = pd.read_csv(logfile)
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=trial_data.keys())

    # Append this trial's data
    trial_row = pd.DataFrame([trial_data])
    results_df = pd.concat([results_df, trial_row], ignore_index=True)

    # Save the updated DataFrame to CSV
    results_df.to_csv(logfile, index=False)
    print(f"Logged trial {trial.number} results to {logfile}")
