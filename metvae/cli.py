import os
import argparse
import pandas as pd
import torch
from .model import MetVAE

def main():
    # ---- Argument parser ----
    parser = argparse.ArgumentParser(description='Run MetVAE correlation + sparsification')

    # Reproducibility & IO
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to abundance CSV (index = samples, columns = features; use --features_as_rows if transposed)')
    parser.add_argument('--features_as_rows', action='store_true',
                        help='If set, features are rows and samples are columns in the input CSV')
    parser.add_argument('--meta', type=str, default=None,
                        help='Optional CSV with sample-level covariates (index must match samples)')
    parser.add_argument('--save_path', type=str, default='./', help='Output directory prefix')

    # Covariates & model size
    parser.add_argument('--continuous_covariate_keys', nargs='+', type=str, default=[],
                        help='Names of continuous covariates in meta')
    parser.add_argument('--categorical_covariate_keys', nargs='+', type=str, default=[],
                        help='Names of categorical covariates in meta')
    
    # Zero-proportion filtering (preprocessing)
    parser.add_argument(
        '--feature_zero_threshold',
        type=float,
        default=0.3,
        help='Drop features with proportion of zeros > threshold. '
             'Set to None (or omit) to keep all features except all-zero.'
    )
    parser.add_argument(
        '--sample_zero_threshold',
        type=float,
        default=None,
        help='Drop samples with proportion of zeros > threshold. '
             'Default None: keep all samples except all-zero.'
    )
    
    # Model architecture
    parser.add_argument("--latent_dim", type=int, default=10, help="Latent dimension")
    parser.add_argument("--hidden_dims", nargs="*", type=int, default=None,
                        help="Encoder hidden layer sizes, e.g. --hidden_dims 256 128. Omit for linear.")
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "tanh", "gelu", "silu", "linear", "none"],
                        help="Activation for MLP layers. 'linear'/'none' means no nonlinearity.")

    # Device & logging
    parser.add_argument('--use_gpu', action='store_true', help='Use CUDA if available')
    parser.add_argument('--logging', action='store_true', help='Enable debug logging')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Max training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Optimizer learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Clip grad norm to this value; set to -1 to disable')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic cuDNN/algorithms (may reduce speed)')

    # Correlation estimation (multiple imputation)
    parser.add_argument('--num_sim', type=int, default=100, help='Number of imputations')
    parser.add_argument('--workers', type=int, default=-1,
                        help='CPU workers for imputation (ignored on GPU); -1 = all cores')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='Correlation sparsity threshold used inside correlation estimation')
    parser.add_argument('--impute_batch_size', type=int, default=100,
                        help='GPU batch size for imputation (only used on CUDA)')

    # Sparsification method
    parser.add_argument('--sparse_method', type=str, default='sec',
                        choices=['pval', 'sec'],
                        help='Choose p-value filtering or SEC algorithm')

    # (A) Filtering options
    parser.add_argument('--p_adj_method', type=str, default='fdr_bh',
                        choices=['bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg',
                                 'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'],
                        help='Multiple testing correction for filtering')
    parser.add_argument('--cutoff', type=float, default=0.05,
                        help='Adjusted p-value cutoff for filtering')

    # (B) SEC options
    parser.add_argument('--rho', type=float, default=-1.0,
                        help='Fixed L1 penalty. Use a negative value (default) to enable CV selection')
    parser.add_argument('--sec_epsilon', type=float, default=1e-5, help='PSD floor epsilon')
    parser.add_argument('--sec_tol', type=float, default=1e-3, help='APG convergence tolerance')
    parser.add_argument('--sec_max_iter', type=int, default=1000, help='Max APG iterations')
    parser.add_argument('--sec_restart', type=int, default=50,
                        help='Nesterov restart period; set <0 to disable')
    parser.add_argument('--no_line_search_apg', action='store_true',
                        help='Disable APG backtracking (line search)')
    parser.add_argument('--sec_delta', type=float, default=None,
                        help='Tiny-entry cutoff δ; None => c_delta*sqrt(log p / n)')
    parser.add_argument('--sec_c_delta', type=float, default=0.1, help='Scale for δ')
    parser.add_argument('--sec_threshold', type=float, default=0.1,
                        help='Final hard threshold applied to SEC result')
    parser.add_argument('--c_grid', nargs='+', type=float,
                        default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                        help='Coarse grid multipliers c for CV: rho = c * sqrt(log(p)/n)')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='K-fold CV splits for rho selection')
    parser.add_argument('--no_refine', action='store_true',
                        help='Disable the single zoom-in refinement after coarse CV')
    parser.add_argument('--refine_points', type=int, default=10, 
                        help='Number of points in the refinement bracket (inclusive)')
    parser.add_argument('--sec_workers', type=int, default=-1,
                        help='CPU workers for CV; -1 = all cores (GPU or <=1 runs sequentially)')
    
    # GraphML export options
    parser.add_argument(
        '--export_graphml',
        action='store_true',
        help='If set, export GraphML correlation networks from the final sparse matrix.'
    )
    parser.add_argument(
        '--graphml_cutoffs',
        nargs='+',
        type=float,
        default=[0.7],
        help='Absolute correlation cutoffs for GraphML export (e.g. 0.9 0.8 0.7 ...).'
    )
    parser.add_argument(
        '--graphml_prefix',
        type=str,
        default='correlation_graph_cutoff',
        help='Filename prefix for GraphML files (suffix = cutoff, extension = .graphml).'
    )

    args = parser.parse_args()

    # ---- Load data ----
    data = pd.read_csv(args.data, index_col=0)
    if args.features_as_rows:
        data = data.T
    meta = None if args.meta is None else pd.read_csv(args.meta, index_col=0)
    
    # ---- Build model ----
    activation = None if args.activation in ("linear", "none") else args.activation
    model = MetVAE(
        data=data,
        features_as_rows=False,
        meta=meta,
        continuous_covariate_keys=args.continuous_covariate_keys,
        categorical_covariate_keys=args.categorical_covariate_keys,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        activation=activation,
        use_gpu=args.use_gpu,
        logging=args.logging,
        feature_zero_threshold=args.feature_zero_threshold,
        sample_zero_threshold=args.sample_zero_threshold,
        seed=args.seed
    )
    
    # ---- Train ----
    model.train(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        max_grad_norm=None if (args.max_grad_norm is None or args.max_grad_norm < 0) else args.max_grad_norm,
        shuffle=True,
        deterministic=args.deterministic
    )
    
    ckpt = {
        "model_state_dict": model.model.state_dict(),
        "optimizer_state_dict": (
            model.optimizer.state_dict()
            if hasattr(model, "optimizer") and getattr(model, "optimizer") is not None
            else None
        ),
        "train_loss": getattr(model, "train_loss", []),
        # Keep raw epoch (if available) and the more useful 'trained_epochs'
        "epoch": getattr(model, "current_epoch", None),
        "trained_epochs": len(getattr(model, "train_loss", [])),
        "learning_rate": args.learning_rate,
    }

    os.makedirs(args.save_path, exist_ok=True)
    torch.save(ckpt, os.path.join(args.save_path, 'model_state.pth'))
    
    # ---- Correlations with multiple imputations ----
    model.get_corr(
        num_sim=args.num_sim,
        workers=args.workers,
        batch_size=args.impute_batch_size,
        threshold=args.threshold,
        seed=args.seed
    )
    
    # ---- Sparsification ----
    if args.sparse_method == 'pval':
        filt = model.sparse_by_p(
            p_adj_method=args.p_adj_method,
            cutoff=args.cutoff
        )
        filt['estimate'].to_csv(os.path.join(args.save_path, 'df_corr.csv'))
        filt['p_value'].to_csv(os.path.join(args.save_path, 'p_values.csv'))
        filt['q_value'].to_csv(os.path.join(args.save_path, 'q_values.csv'))
        filt['sparse_estimate'].to_csv(os.path.join(args.save_path, 'df_sparse_pval.csv'))
    else:  # SEC
        rho_val = None if args.rho is None or args.rho < 0 else float(args.rho)
        restart_val = None if (args.sec_restart is None or args.sec_restart < 0) else int(args.sec_restart)
        filt = model.sparse_by_sec(
            rho=rho_val,
            epsilon=args.sec_epsilon,
            tol=args.sec_tol,
            max_iter=args.sec_max_iter,
            restart=restart_val,
            line_search_apg=not args.no_line_search_apg,
            delta=args.sec_delta,
            n_samples=None,
            c_delta=args.sec_c_delta,
            threshold=args.sec_threshold,
            c_grid=args.c_grid,
            n_splits=args.n_splits,
            seed=args.seed,
            workers=args.sec_workers,
            refine=(not args.no_refine),
            refine_points=args.refine_points
        )
        # Save outputs
        filt['estimate'].to_csv(os.path.join(args.save_path, 'df_corr.csv'))
        filt['sparse_estimate'].to_csv(os.path.join(args.save_path, 'df_sparse_sec.csv'))
        # Optional extras
        if filt.get('best_rho') is not None:
            with open(os.path.join(args.save_path, 'sec_selected.txt'), 'w') as f:
                f.write(f"best_rho={filt['best_rho']}\n")
        if filt.get('scores_by_rho') is not None:
            filt['scores_by_rho'].to_csv(os.path.join(args.save_path, 'sec_scores.csv'), index=False)
            
    # ---- GraphML export ----
    if args.export_graphml:
        model.export_graphml(
            sparse_df=filt['sparse_estimate'],
            cutoffs=args.graphml_cutoffs,
            output_dir=args.save_path,
            file_prefix=args.graphml_prefix,
        )

if __name__ == '__main__':
    main()
