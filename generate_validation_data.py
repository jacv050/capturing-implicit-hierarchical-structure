import sys
from pvae.datasets.youtube_dataset import YoutubeDataset
sys.path.append(".")
sys.path.append("..")
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--output_dir', type=str, default='../dataset/p1/validation')
parser.add_argument('--annotation', type=str, default='../dataset_info/finegym/annotations/finegym_annotation_info_v1.1.json')
args = parser.parse_args()

if __name__ == "__main__":
    dataset = YoutubeDataset(args.annotation)
    dataset.save_by_video(args.output_dir)