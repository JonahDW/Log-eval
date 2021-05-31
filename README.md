### Log-eval

This is a module for use inside the cluster at IUCAA, on data that has been processed through ARTIP. It reads logs of different processing steps and extracts useful information to assess the quality of the processing. 

'''
usage: read_logs.py [-h] [-s SELECTION] dataset_path experiment

positional arguments:
  dataset_path          Path to data.
  experiment            Experiment index.

optional arguments:
  -h, --help            show this help message and exit
  -s SELECTION, --selection SELECTION
                        Timerange of logs to select. Use 'all' to select all
                        logs, 'last' to only select most recent logs for each
                        run (default = 'all').
'''
