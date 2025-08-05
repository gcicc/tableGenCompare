# Hyperparmeters in Synthetic Table Generation Models

# CTGAN
import optuna

# CTGAN Optuna Search Space
def ctgan_search_space(trial):
    return {
        "epochs": trial.suggest_int("epochs", 50, 500, step=50),  # Number of training cycles
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),  # Divisible by pac
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),  # LR for both generator and discriminator
        "generator_dim": trial.suggest_categorical(  # Architecture for generator
            "generator_dim", [[256, 256], [256, 128, 64], [512, 256, 128]]
        ),
        "discriminator_dim": trial.suggest_categorical(  # Architecture for discriminator
            "discriminator_dim", [[256, 256], [256, 128, 64], [512, 256, 128]]
        ),
        "pac": trial.suggest_int("pac", 5, 20),  # Packed samples for discriminator
        "embedding_dim": trial.suggest_int("embedding_dim", 64, 256, step=32),  # Input noise vector size
        "generator_decay": trial.suggest_loguniform("generator_decay", 1e-6, 1e-3),  # Generator weight decay
        "discriminator_decay": trial.suggest_loguniform("discriminator_decay", 1e-6, 1e-3),  # Discriminator weight decay
        "discriminator_steps": trial.suggest_int("discriminator_steps", 1, 5),  # Discriminator steps per generator update
        "log_frequency": trial.suggest_categorical("log_frequency", [True, False]),  # Log-based conditional sampling
        "conditional_generation": trial.suggest_categorical("conditional_generation", [True, False]),  # Conditional generation
        "min_max_enforce": trial.suggest_categorical("min_max_enforce", [True, False]),  # Enforce min/max values
    }



import optuna

# Objective Function for CTGAN
def objective(trial):
    params = ctgan_search_space(trial)
    # Initialize your CTGAN instance with suggested parameters
    ctgan = CTGAN(
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        generator_lr=params["learning_rate"],
        generator_dim=params["generator_dim"],
        discriminator_dim=params["discriminator_dim"],
        pac=params["pac"],
        embedding_dim=params["embedding_dim"],
        generator_decay=params["generator_decay"],
        discriminator_decay=params["discriminator_decay"],
        discriminator_steps=params["discriminator_steps"],
        log_frequency=params["log_frequency"],
        conditional_generation=params["conditional_generation"],
        min_max_enforce=params["min_max_enforce"],
    )

    # Fit the CTGAN model
    ctgan.fit(real_data)  # Replace `real_data` with your dataset

    # Generate synthetic data and evaluate using a scoring function
    synthetic_data = ctgan.sample(num_samples=len(real_data))
    score = evaluate_synthetic_data(real_data, synthetic_data)  # Define your utility/quality metric

    return score

# Run the Optuna Study
study = optuna.create_study(direction="maximize")  # Optimize for highest score
study.optimize(objective, n_trials=50)  # Adjust the number of trials as needed

# Print Best Parameters
print("Best trial:")
trial = study.best_trial

print(f"Score: {trial.value}")
print("Best hyperparameters: ", trial.params)

# TVAE

import optuna

# TVAE Optuna Search Space
def tvae_search_space(trial):
    return {
        "epochs": trial.suggest_int("epochs", 50, 500, step=50),  # Training cycles
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),  # Training batch size
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),  # Learning rate
        "compress_dims": trial.suggest_categorical(  # Encoder architecture
            "compress_dims", [[128, 128], [256, 128], [256, 128, 64]]
        ),
        "decompress_dims": trial.suggest_categorical(  # Decoder architecture
            "decompress_dims", [[128, 128], [64, 128], [64, 128, 256]]
        ),
        "embedding_dim": trial.suggest_int("embedding_dim", 32, 256, step=32),  # Latent space bottleneck size
        "l2scale": trial.suggest_loguniform("l2scale", 1e-6, 1e-2),  # L2 regularization weight
        "dropout": trial.suggest_uniform("dropout", 0.0, 0.5),  # Dropout probability
        "log_frequency": trial.suggest_categorical("log_frequency", [True, False]),  # Use log frequency for representation
        "conditional_generation": trial.suggest_categorical("conditional_generation", [True, False]),  # Conditioned generation
    }


import optuna

# Objective Function for TVAE
def objective(trial):
    params = tvae_search_space(trial)
    
    # Initialize TVAE instance with suggested parameters
    tvae = TVAE(
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        compress_dims=params["compress_dims"],
        decompress_dims=params["decompress_dims"],
        embedding_dim=params["embedding_dim"],
        l2scale=params["l2scale"],
        log_frequency=params["log_frequency"],
        conditional_generation=params["conditional_generation"]
    )

    # Fit TVAE to real data
    tvae.fit(real_data)  # Replace `real_data` with your dataset

    # Generate synthetic data and evaluate using a scoring function
    synthetic_data = tvae.sample(num_samples=len(real_data))
    score = evaluate_synthetic_data(real_data, synthetic_data)  # Define your evaluation metric

    return score

# Run the Optuna Study
study = optuna.create_study(direction="maximize")  # Optimize for the highest score
study.optimize(objective, n_trials=50)  # Adjust the number of trials as needed

# Print Best Parameters
print("Best trial:")
trial = study.best_trial

print(f"Score: {trial.value}")
print("Best hyperparameters: ", trial.params)

# CoupluaGAN
import optuna

# CopulaGAN Optuna Search Space
def copulagan_search_space(trial):
    return {
        "epochs": trial.suggest_int("epochs", 50, 500, step=50),  # Training cycles
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),  # Training batch size
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),  # Learning rate
        "generator_dim": trial.suggest_categorical(  # Generator architecture
            "generator_dim", [[256, 256], [256, 128, 64], [512, 256, 128]]
        ),
        "discriminator_dim": trial.suggest_categorical(  # Discriminator architecture
            "discriminator_dim", [[256, 256], [256, 128, 64], [512, 256, 128]]
        ),
        "pac": trial.suggest_int("pac", 5, 20),  # Packed samples for discriminator
        "embedding_dim": trial.suggest_int("embedding_dim", 64, 256, step=32),  # Input noise vector size
        "generator_decay": trial.suggest_loguniform("generator_decay", 1e-6, 1e-3),  # Generator weight decay
        "discriminator_decay": trial.suggest_loguniform("discriminator_decay", 1e-6, 1e-3),  # Discriminator weight decay
        "discriminator_steps": trial.suggest_int("discriminator_steps", 1, 5),  # Discriminator steps per generator update
        "log_frequency": trial.suggest_categorical("log_frequency", [True, False]),  # Log-based conditional sampling
        "conditional_generation": trial.suggest_categorical("conditional_generation", [True, False]),  # Conditional generation
        "min_max_enforce": trial.suggest_categorical("min_max_enforce", [True, False]),  # Enforce min/max values
    }

import optuna

# Objective Function for CopulaGAN
def objective(trial):
    params = copulagan_search_space(trial)
    
    # Initialize CopulaGAN instance with suggested parameters
    copulagan = CopulaGAN(
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        generator_lr=params["learning_rate"],
        generator_dim=params["generator_dim"],
        discriminator_dim=params["discriminator_dim"],
        pac=params["pac"],
        embedding_dim=params["embedding_dim"],
        generator_decay=params["generator_decay"],
        discriminator_decay=params["discriminator_decay"],
        discriminator_steps=params["discriminator_steps"],
        log_frequency=params["log_frequency"],
        conditional_generation=params["conditional_generation"],
        min_max_enforce=params["min_max_enforce"]
    )

    # Fit CopulaGAN to real data
    copulagan.fit(real_data)  # Replace `real_data` with your dataset

    # Generate synthetic data and evaluate using a scoring function
    synthetic_data = copulagan.sample(num_samples=len(real_data))
    score = evaluate_synthetic_data(real_data, synthetic_data)  # Define your evaluation metric

    return score

# Run the Optuna Study
study = optuna.create_study(direction="maximize")  # Optimize for highest score
study.optimize(objective, n_trials=50)  # Adjust number of trials as needed

# Print Best Parameters
print("Best trial:")
trial = study.best_trial

print(f"Score: {trial.value}")
print("Best hyperparameters: ", trial.params)


# GANerAid

import optuna

# GANerAID Optuna Search Space
def ganeraid_search_space(trial):
    return {
        "epochs": trial.suggest_int("epochs", 50, 500, step=50),  # Training iterations
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),  # Training batch size
        "lr_g": trial.suggest_loguniform("lr_g", 1e-5, 1e-2),  # Generator learning rate
        "lr_d": trial.suggest_loguniform("lr_d", 1e-5, 1e-2),  # Discriminator learning rate
        "hidden_feature_space": trial.suggest_int("hidden_feature_space", 50, 512, step=50),  # Latent feature space
        "nr_of_rows": trial.suggest_int("nr_of_rows", 10, 50, step=5),  # Generated rows per training cycle
        "binary_noise": trial.suggest_uniform("binary_noise", 0.0, 0.5),  # Noise added to binary features
        "generator_decay": trial.suggest_loguniform("generator_decay", 1e-6, 1e-3),  # Generator weight decay
        "discriminator_decay": trial.suggest_loguniform("discriminator_decay", 1e-6, 1e-3),  # Discriminator weight decay
        "dropout_generator": trial.suggest_uniform("dropout_generator", 0.0, 0.5),  # Dropout in generator layers
        "dropout_discriminator": trial.suggest_uniform("dropout_discriminator", 0.0, 0.5),  # Dropout in discriminator layers
    }

import optuna

# Objective Function for GANerAID
def objective(trial):
    params = ganeraid_search_space(trial)
    
    # Initialize GANerAID with suggested parameters
    ganeraid = GANerAID(
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        lr_g=params["lr_g"],
        lr_d=params["lr_d"],
        hidden_feature_space=params["hidden_feature_space"],
        nr_of_rows=params["nr_of_rows"],
        binary_noise=params["binary_noise"],
        generator_decay=params["generator_decay"],
        discriminator_decay=params["discriminator_decay"],
        dropout_generator=params["dropout_generator"],
        dropout_discriminator=params["dropout_discriminator"],
    )

    # Fit GANerAID to real data
    ganeraid.fit(real_data)  # Replace `real_data` with your dataset

    # Generate synthetic data and evaluate using a scoring function
    synthetic_data = ganeraid.sample(len(real_data))  # Generate synthetic data
    score = evaluate_synthetic_data(real_data, synthetic_data)  # Define your custom evaluation metric

    return score

# Run the Optuna Study
study = optuna.create_study(direction="maximize")  # Optimize for the highest score
study.optimize(objective, n_trials=50)  # Adjust the number of trials as needed

# Print Best Parameters
print("Best trial:")
trial = study.best_trial

print(f"Score: {trial.value}")
print("Best hyperparameters: ", trial.params)

# TableGAN

import optuna

# TableGAN Optuna Search Space
def tablegan_search_space(trial):
    return {
        "epochs": trial.suggest_int("epochs", 50, 500, step=50),  # Training iterations
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),  # Batch size
        "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),  # Learning rate for Adam optimizer
        "beta1": trial.suggest_uniform("beta1", 0.3, 0.9),  # Beta1 for Adam optimizer
        "beta2": trial.suggest_uniform("beta2", 0.8, 0.999),  # Beta2 for Adam optimizer
        "generator_dim": trial.suggest_categorical(  # Generator architecture
            "generator_dim", [[256, 128], [512, 256, 128], [128, 128]]
        ),
        "discriminator_dim": trial.suggest_categorical(  # Discriminator architecture
            "discriminator_dim", [[256, 128], [512, 256, 128], [128, 128]]
        ),
        "dropout_generator": trial.suggest_uniform("dropout_generator", 0.0, 0.5),  # Dropout in generator
        "dropout_discriminator": trial.suggest_uniform("dropout_discriminator", 0.0, 0.5),  # Dropout in discriminator
        "mlp_activation": trial.suggest_categorical("mlp_activation", ["relu", "leakyrelu", "tanh"]),  # MLP activation type
    }

# Objective Function for TableGAN
def objective(trial):
    params = tablegan_search_space(trial)
    
    # Initialize TableGAN with trial parameters
    tablegan = TableGAN(
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        lr=params["lr"],
        beta1=params["beta1"],
        beta2=params["beta2"],
        generator_dim=params["generator_dim"],
        discriminator_dim=params["discriminator_dim"],
        dropout_generator=params["dropout_generator"],
        dropout_discriminator=params["dropout_discriminator"],
        mlp_activation=params["mlp_activation"],
    )

    # Fit TableGAN to real data
    tablegan.fit(real_data)  # Replace `real_data` with your dataset

    # Generate synthetic data and evaluate
    synthetic_data = tablegan.sample(len(real_data))  # Generate synthetic samples
    score = evaluate_synthetic_data(real_data, synthetic_data)  # Define your evaluation metric

    return score

# Run Optuna Study
study = optuna.create_study(direction="maximize")  # Optimize for the highest score
study.optimize(objective, n_trials=50)  # Adjust the number of trials as needed

# Output best parameters
print(f"Best trial score: {study.best_trial.value}")
print(f"Best hyperparameters: {study.best_trial.params}")