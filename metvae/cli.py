import argparse
import random
import numpy as np
import pandas as pd
import torch
from .model import MetIWAE, MetVAE

def main():
    # Create the argument parser
    arg_parser = argparse.ArgumentParser(description='Run MetVAE')

    # Define arguments
    arg_parser.add_argument('--seed', type=int, help='Set a random seed for reproducibility', default=123)
    arg_parser.add_argument('--model', type=str, help='Select either the VAE or IWAE model', default='VAE')
    arg_parser.add_argument('--data', type=str, help='Path to the metabolic abundance file (CSV). The index column should be the first column.', required=True)
    arg_parser.add_argument('--features_as_rows', action='store_true', help='Use if features are in rows and samples in columns (default: features in columns)')
    arg_parser.add_argument('--meta', type=str, help='Path to the sample meta file (CSV). The index column should be the first column.', default=None)
    arg_parser.add_argument('--continuous_covariate_keys', nargs='+', type=str, help='Names of continuous covariates', default=[])
    arg_parser.add_argument('--categorical_covariate_keys', nargs='+', type=str, help='Names of categorical covariates', default=[])
    arg_parser.add_argument('--latent_dim', type=int, help='Set the dimension of laten variables', default=100)
    arg_parser.add_argument('--num_importance_samples', type=int, help='Set the number of importance samples to generate.', default=1)
    arg_parser.add_argument('--use_gpu', action='store_true', help='Enable GPU acceleration (default: CPU only)')
    arg_parser.add_argument('--logging', action='store_true', help='Enable debug logging (default: disabled)')
    arg_parser.add_argument('--batch_size', type=int, help='Number of samples per batch during training', default=32)
    arg_parser.add_argument('--num_workers', type=int, help='Number of worker threads for data loading', default=0)
    arg_parser.add_argument('--max_epochs', type=int, help='Maximum number of training epochs', default=1000)
    arg_parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer', default=0.001)
    arg_parser.add_argument('--th_len', type=int, help='Number of threshold values to test', default=30)
    arg_parser.add_argument('--n_cv', type=int, help='Number of cross-validation folds', default=5)
    arg_parser.add_argument('--soft', action='store_true', help='Use soft thresholding instead of hard (default: hard)')
    arg_parser.add_argument('--alpha_grid', nargs='+', type=float, help='Custom grid of sparsity penalty parameters to test', default=[0.0])
    arg_parser.add_argument('--n_jobs', type=int, help='The maximum number of concurrently running jobs', default=None)
    arg_parser.add_argument('--save_path', type=str, help='Path to save the outputs', default='./')

    # Parse the arguments
    args = arg_parser.parse_args()

    # Main code
    # Input data
    data = pd.read_csv(args.data, index_col=0)

    if args.features_as_rows:
        data = data.T
    
    feature_name = data.columns.tolist()

    if args.meta is None:
        meta = None
    else:
        meta = pd.read_csv(args.meta, index_col=0)

    # Run the VAE/IWAE model
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.model == 'VAE':
        model = MetVAE(data=data,
                       meta=meta,
                       continuous_covariate_keys=args.continuous_covariate_keys,
                       categorical_covariate_keys=args.categorical_covariate_keys,
                       latent_dim=args.latent_dim,
                       use_gpu=args.use_gpu,
                       logging=args.logging)
    else:
        model = MetIWAE(data=data,
                        meta=meta,
                        continuous_covariate_keys=args.continuous_covariate_keys,
                        categorical_covariate_keys=args.categorical_covariate_keys,
                        latent_dim=args.latent_dim,
                        num_importance_samples=args.num_importance_samples,
                        use_gpu=args.use_gpu,
                        logging=args.logging)
    
    model.train(batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_epochs=args.max_epochs,
                learning_rate=args.learning_rate,
                log_every_n_steps=1)

    # Save the model state
    torch.save(model.model.state_dict(), args.save_path + 'model_state.pth')
    
    # Obtain correlations
    model.get_corr(num_sim=1000)
    random.seed(args.seed)
    results = model.sparse_by_thresholding(th_len=args.th_len, 
                                           n_cv=args.n_cv, 
                                           soft=args.soft, 
                                           alpha_grid=np.array(args.alpha_grid),
                                           n_jobs=args.n_jobs)
    est_cor = results['sparse_estimate']
    
    # Save the correlations
    df_cor = pd.DataFrame(
        est_cor,
        index=feature_name,
        columns=feature_name
    )
    df_cor.to_csv(args.save_path + 'df_corr.csv')
    
if __name__ == '__main__':
    main()
