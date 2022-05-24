"""
Evaluate barge in metric from VAD tags

Usage:
  eval_barge_in.py <true-labels> <pred-labels> [--vad-yaml=<vad_yaml_path>]

Options:
  --vad-yaml=<vad_yaml_path>    Path to params for evaluating vad metrics [default: assets/vad.yaml]

Arguments:
  <true-labels>             Path to file with true labels with our dataframe
                            definitions.
  <pred-labels>             Path to file with predicted labels with our
                            dataframe definitions.
"""

import pandas as pd
from docopt import docopt

from eevee.metrics.vad import barge_in_report
from eevee.utils import parse_yaml


if __name__ == "__main__":

	args = docopt(__doc__)
	true_labels = pd.read_csv(args["<true-labels>"])
	pred_labels = pd.read_csv(args["<pred-labels>"])
	vad_params = parse_yaml(args["--vad-yaml"])

	output = barge_in_report(true_labels, pred_labels, vad_params)
	print(output)
