import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Arguments for training and GAN")
    
    parser.add_argument('--data_dir', default='dataset/hgg_data')
    parser.add_argument('--plot', action='store_true')



    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--device', default='GPU')

    parser.add_argument('--resume', action='store_true', help="Resume training from a checkpoint")

    parser.add_argument('--TCIA_data', default='dataset/TCIA', type=str)

    parser.add_argument('--vol', default='', type=str)


    args = parser.parse_args()
    return args
