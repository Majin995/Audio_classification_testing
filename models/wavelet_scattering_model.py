import torch
import torch.nn as nn
import pytorch_lightning as pl
from kymatio.torch import Scattering1D
from torchmetrics import (
    MetricCollection,
    F1Score,
    FBetaScore,
    MatthewsCorrCoef,
    ConfusionMatrix,
    ROC,
    AUROC
)


# --- NEW IMPORTS FOR NAS ---
try:
    import optuna
    from optuna.integration import PyTorchLightningPruningCallback
except ImportError:
    print("Optuna is not installed. Please install it with 'pip install optuna optuna-dashboard' to use the NAS features.")
    optuna = None

from torch.utils.data import TensorDataset, DataLoader
# --- END NEW IMPORTS ---


class WaveletScatteringClassifier(pl.LightningModule):
    """
    A PyTorch Lightning model for audio classification using Kymatio's Wavelet Scattering Transform.
    This version is adapted for Neural Architecture Search (NAS) with Optuna.

    To use hardware logging (GPU, CPU, etc.), you need to enable the appropriate callbacks
    in your PyTorch Lightning Trainer. For example:
    from pytorch_lightning.callbacks import DeviceStatsMonitor
    trainer = pl.Trainer(callbacks=[DeviceStatsMonitor()])

    Args:
        input_size (int): The size (length) of the input audio samples.
        num_classes (int): The number of classes for classification.
        J (int): The number of scales for the scattering transform (log2 of the max scale).
        Q (int): The number of wavelets per octave.
        lr (float): The learning rate for the optimizer.
        f_beta (float): The beta value for the FBetaScore metric.
        hidden_dims (list[int]): A list of hidden layer dimensions for the classifier.
    """
    def __init__(self, input_size: int, num_classes: int, J: int = 8, Q: int = 1, lr: float = 1e-3, f_beta: float = 1.0, hidden_dims: list = [256], optuna_trial: 'optuna.trial.Trial' = None):
        super().__init__()
        self.save_hyperparameters()

        # 1. Scattering Transform Layer
        self.scattering = Scattering1D(J=self.hparams.J, shape=(self.hparams.input_size,), Q=self.hparams.Q)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.hparams.input_size)
            scattering_coeffs = self.scattering(dummy_input)
            scattering_output_dim = scattering_coeffs.shape[1]

        # 2. Dynamic Classifier Head for NAS
        layers = []
        in_features = scattering_output_dim
        for h_dim in self.hparams.hidden_dims:
            layers.append(nn.Linear(in_features, h_dim))
            layers.append(nn.ReLU())
            in_features = h_dim
        layers.append(nn.Linear(in_features, self.hparams.num_classes))
        self.classifier = nn.Sequential(*layers)

        # 3. Metrics from TorchMetrics
        task = "multiclass" if num_classes > 2 else "binary"
        
        metrics = MetricCollection({
            'f1_score': F1Score(task=task, num_classes=num_classes),
            'f_beta': FBetaScore(task=task, num_classes=num_classes, beta=self.hparams.f_beta),
            'mcc': MatthewsCorrCoef(task=task, num_classes=num_classes)
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        self.val_roc = ROC(task=task, num_classes=num_classes)
        self.test_roc = ROC(task=task, num_classes=num_classes)
        self.val_auroc = AUROC(task=task, num_classes=num_classes)
        self.test_auroc = AUROC(task=task, num_classes=num_classes)

        self.val_confusion_matrix = ConfusionMatrix(task=task, num_classes=num_classes)
        self.test_confusion_matrix = ConfusionMatrix(task=task, num_classes=num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        
        scattering_coeffs = self.scattering(x)
        pooled_coeffs = torch.mean(scattering_coeffs, dim=-1)
        logits = self.classifier(pooled_coeffs)
        return logits

    def _common_step(self, batch, batch_idx, stage: str):
        x, y = batch
        logits = self.forward(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        metrics = getattr(self, f'{stage}_metrics')
        metrics.update(logits, y)

        if stage == 'val':
            self.val_roc.update(logits, y)
            self.val_auroc.update(logits, y)
            self.val_confusion_matrix.update(logits, y)
        elif stage == 'test':
            self.test_roc.update(logits, y)
            self.test_auroc.update(logits, y)
            self.test_confusion_matrix.update(logits, y)

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'test')
        
    def _on_common_epoch_end(self, stage: str):
        metrics = getattr(self, f'{stage}_metrics')
        self.log_dict(metrics.compute(), on_step=False, on_epoch=True)
        metrics.reset()

        if stage == 'val' and self.logger:
            fig, _ = self.val_roc.plot(score=True)
            #self.logger.experiment.add_figure('val_roc_curve', fig, self.current_epoch)
            self.val_roc.reset()
            
            self.log('val_auroc', self.val_auroc.compute(), on_epoch=True)
            self.val_auroc.reset()

            fig, _ = self.val_confusion_matrix.plot()
            #self.logger.experiment.add_figure('val_confusion_matrix', fig, self.current_epoch)
            self.val_confusion_matrix.reset()

    def on_train_epoch_end(self):
        self._on_common_epoch_end('train')

    def on_validation_epoch_end(self):
        self._on_common_epoch_end('val')

        if self.hparams.optuna_trial is not None:
            val_loss = self.trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                self.hparams.optuna_trial.report(val_loss, self.current_epoch)
                if self.hparams.optuna_trial.should_prune():
                    raise optuna.TrialPruned()

    def on_test_epoch_end(self):
        # Implementation for test epoch end remains similar, logging to epoch 0
        self._on_common_epoch_end('test')
        if self.logger:
            fig, _ = self.test_roc.plot(score=True)
            #self.logger.experiment.add_figure('test_roc_curve', fig, 0)
            self.test_roc.reset()
            
            self.log('test_auroc', self.test_auroc.compute())
            self.test_auroc.reset()
            
            fig, _ = self.test_confusion_matrix.plot()
            #self.logger.experiment.add_figure('test_confusion_matrix', fig, 0)
            self.test_confusion_matrix.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

if __name__ == '__main__':
    if optuna is None:
        print("Cannot run NAS example because Optuna is not installed.")
    else:
        # --- Optuna NAS Example ---
        # This example demonstrates how to use Optuna to find the best architecture
        # and hyperparameters for the WaveletScatteringClassifier.

        # 1. Define constants and dummy data
        INPUT_SIZE = 8192   # Use a smaller input size for faster trials
        NUM_CLASSES = 10
        N_TRAIN_SAMPLES = 200
        N_VAL_SAMPLES = 50
        BATCH_SIZE = 16
        N_EPOCHS = 5

        # Create dummy data that is vaguely realistic (0-1 range)
        train_x = torch.rand(N_TRAIN_SAMPLES, INPUT_SIZE)
        train_y = torch.randint(0, NUM_CLASSES, (N_TRAIN_SAMPLES,))
        val_x = torch.rand(N_VAL_SAMPLES, INPUT_SIZE)
        val_y = torch.randint(0, NUM_CLASSES, (N_VAL_SAMPLES,))

        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=BATCH_SIZE)
        
        # 2. Define the Objective function for Optuna
        def objective(trial: optuna.trial.Trial) -> float:
            """
            This function is called by Optuna for each trial. It defines the search space,
            trains a model, and returns a metric for Optuna to optimize.
            """
            # -- Hyperparameter Search Space --
            # a) Scattering parameters
            J = trial.suggest_int("J", 4, 8)
            Q = trial.suggest_int("Q", 1, 4)

            # b) Optimizer parameters
            lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

            # c) Classifier architecture
            n_layers = trial.suggest_int("n_layers", 1, 3)
            hidden_dims = []
            for i in range(n_layers):
                out_features = trial.suggest_int(f"n_units_l{i}", 32, 512, log=True)
                hidden_dims.append(out_features)

            # -- Model and Trainer Setup --
            model = WaveletScatteringClassifier(
                input_size=INPUT_SIZE,
                num_classes=NUM_CLASSES,
                J=J,
                Q=Q,
                lr=lr,
                hidden_dims=hidden_dims
            )
            
            # Add a callback for pruning unpromising trials
            pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

            trainer = pl.Trainer(
                max_epochs=N_EPOCHS,
                accelerator="auto", # Automatically uses GPU if available
                devices=1,
                callbacks=[pruning_callback],
                logger=True, # Enables default TensorBoardLogger
                enable_progress_bar=False, # Disable progress bar for cleaner logs
                enable_model_summary=False,
            )
            
            # Log the number of parameters to Optuna
            trial.set_user_attr("n_params", sum(p.numel() for p in model.parameters() if p.requires_grad))

            # Train the model
            trainer.fit(model, train_loader, val_loader)

            # Return the metric to be optimized (e.g., final validation loss)
            return trainer.callback_metrics["val_loss"].item()

        # 3. Create and run the Optuna study
        N_TRIALS = 25 # Number of different architectures to test
        
        # We want to minimize the validation loss
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPEASampler(), pruner=optuna.pruners.MedianPruner())
        
        print(f"Running NAS for {N_TRIALS} trials... This may take a while.")
        study.optimize(objective, n_trials=N_TRIALS, timeout=600) # 10-minute timeout

        # 4. Print results
        print("\n--- NAS Study Complete ---")
        print(f"Number of finished trials: {len(study.trials)}")

        best_trial = study.best_trial
        print("Best trial:")
        print(f"  Value (min val_loss): {best_trial.value:.4f}")
        
        print("  Best Architecture & Hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        
        print(f"  Number of parameters: {best_trial.user_attrs['n_params']}")

        # You can also visualize the results using optuna-dashboard
        # Run in your terminal: optuna-dashboard sqlite:///example-study.db
        # study.storage is a string, so we need to save to a file first.
        # This example uses in-memory storage. For persistence, use:
        # storage = "sqlite:///nas_study.db"
        # study = optuna.create_study(storage=storage, ...)
        print("\nTo visualize results, re-un with a persistent storage backend like SQLite.")
