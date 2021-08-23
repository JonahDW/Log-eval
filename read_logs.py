import os
import re
import sys
import glob
import json

from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

import numpy as np
from astropy.table import Table, unique

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import init_logger
import rms_map

logger = init_logger.setup_logger('log-eval.log')

def match_format_string(format_str, s):
    '''
    Match s against the given format string, return dict of matches.

    We assume all of the arguments in format string are named keyword arguments (i.e. no {} or
    {:0.2f}). We also assume that all chars are allowed in each keyword argument, so separators
    need to be present which aren't present in the keyword arguments (i.e. '{one}{two}' won't work
    reliably as a format string but '{one}-{two}' will if the hyphen isn't used in {one} or {two}).

    We raise if the format string does not match s.
    Source: https://stackoverflow.com/questions/10663093/use-python-format-string-in-reverse-for-parsing

    Example:
    fs = '{test}-{flight}-{go}'
    s = fs.format('first', 'second', 'third')
    match_format_string(fs, s) -> {'test': 'first', 'flight': 'second', 'go': 'third'}
    '''

    # First split on any keyword arguments, note that the names of keyword arguments will be in the
    # 1st, 3rd, ... positions in this list
    tokens = re.split(r'\{(.*?)\}', format_str)
    keywords = tokens[1::2]

    # Now replace keyword arguments with named groups matching them. We also escape between keyword
    # arguments so we support meta-characters there. Re-join tokens to form our regexp pattern
    tokens[1::2] = map(u'(?P<{}>.*)'.format, keywords)
    tokens[0::2] = map(re.escape, tokens[0::2])
    pattern = ''.join(tokens)

    # Use our pattern to match the given string, raise if it doesn't match
    matches = re.match(pattern, s)
    if not matches:
        raise Exception("Format string did not match")

    # Return a dict with all of our keywords and their values
    return {x: matches.group(x) for x in keywords}

def parse_file(filename):
    '''
    Read a file and return a list of its lines
    '''
    with open(filename) as f:
        file = f.read()
        lines = file.split('\n')

    return lines

class Logs:

    def __init__(self, card, output):
        self.output = output
        self.card = card
        self.software_version = ''

        # Identify the artip block
        if 'cont' in card:
            mspath = glob.glob(os.path.join(card,'output','*.ms'))[0].rsplit('/',1)
            self.card_name = card.split('/')[-1]
            self.output_path = mspath[0]
            self.dataset = mspath[1].split('_')[0]

        if 'cal' in card:
            card_split = card.rsplit('/',2)
            mspath = glob.glob(os.path.join(card_split[0],'*.ms'))[0].rsplit('/',1)
            self.card_name = card_split[2]
            self.output_path = os.path.join(card,'output')
            self.dataset = 'ABSDATA32K'

        if 'cube' in card:
            logger.error('Functionality for this card is not yet implemented')
            self.card_name = None

        # Initialize data structures
        self.casa_log = Table()
        self.casa_error = Table()
        self.artip_error = Table()

        self.casa_warn = []
        self.casa_severe = []
        self.artip_errors = []

    def index_logs(self, select='last'):
        '''
        Build up tables and index all the logs present in a card

        Keyword arguments:
        select -- Which logs to index, choice currently between 'all'
                  selecting all and 'last' selecting only the most recent
                  run of each process
        '''
        self.casa_log['file'] = glob.glob(os.path.join(self.output_path,self.dataset,'casa_log','*'))
        self.casa_error['file'] = glob.glob(os.path.join(self.output_path,self.dataset,'casa_error','*'))
        self.artip_error['file'] = glob.glob(os.path.join(self.output_path,'*error*'))

        # Parse the timestamps from filename
        self.casa_log['time'] = [datetime.strptime(time.split('_')[-1].split('.')[0], '%H-%M-%S--%d-%m-%Y')
                            for time in self.casa_log['file']]
        self.casa_error['time'] = [datetime.strptime(time.split('_')[-1].split('.')[0], '%H-%M-%S--%d-%m-%Y')
                              for time in self.casa_error['file']]
        self.artip_error['time'] = [datetime.strptime(time.split('_')[-1].split('.')[0], '%H:%M:%S-%d-%m-%Y')
                               for time in self.artip_error['file']]

        # Parse task from filename
        self.casa_log['task'] = [os.path.basename(file).split('_')[2] for file in self.casa_log['file']]
        self.casa_error['task'] = [os.path.basename(file).split('_')[2] for file in self.casa_error['file']]

        # If no logs found, go back
        if len(self.casa_log) == 0 and len(self.casa_error) == 0:
            logger.warning('No logs found in card')
            return False

        # Sort in time
        self.casa_log.sort('time')
        self.casa_log.reverse()

        self.casa_error.sort('time')
        self.casa_error.reverse()

        self.artip_error.sort('time')
        self.artip_error.reverse()

        # Select most recent for each task
        if select == 'last':
            self.casa_log = unique(self.casa_log, keys='task')
            self.casa_error = unique(self.casa_error, keys='task')
            self.artip_error = self.artip_error[0]

        if select == 'all':
            pass

        # Retrieve software version
        logfile = parse_file(self.casa_log['file'][0])
        for line in logfile:
            if 'CASA Version' in line:
                self.software_version = line.split('\t')[-1].strip()
                break

        return True

    def get_processing_time(self):
        '''
        Read output file in order to get processing time
        '''
        std_out = glob.glob(os.path.join(self.output_path,'std_out_*'))

        out_time = [datetime.strptime(os.path.basename(time).split('_',2)[2].split('.')[0], '%H_%M_%S__%Y_%m_%d')
                            for time in std_out]
        sorted_idx = sorted(range(len(out_time)), key=out_time.__getitem__)
        lines = parse_file(std_out[sorted_idx[-1]])

        start_line = re.sub(r'^.*?202', '202', lines[1])
        end_line = re.sub(r'^.*?202', '202', lines[-2])

        start_time = datetime.strptime(start_line.split('    ')[0], '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(end_line.split('    ')[0], '%Y-%m-%d %H:%M:%S')
        diff = end_time - start_time

        time_seconds = diff.total_seconds()
        logger.info(f'Time taken by pipeline: {int(time_seconds/3600)} hours and {int(time_seconds/60 % 60)} minutes')

    def read_errors(self):
        '''
        Get all error messages from the indexed logs
        '''
        # Parse CASA errors
        for file in self.casa_error['file']:
            lines = parse_file(file)
            filename = os.path.basename(file)
            if len(lines) == 0:
                continue
            for line in lines:
                if 'WARN' in line:
                    self.casa_warn.append([line, filename, self.software_version])
                if 'SEVERE' in line:
                    self.casa_severe.append([line, filename, self.software_version])

        # Do this otherwise we get rekt
        files = self.artip_error['file']
        files = [files] if isinstance(files, str) else files

        # Parse all other errors
        if len(self.artip_error) > 0:
            for file in files:
                lines = parse_file(file)
                filename = os.path.basename(file)
                if len(lines) <= 1:
                    continue
                elif any('Traceback' in l for l in lines):
                    self.artip_errors.append([lines, filename])

        # Make into tables to make stuff easier
        self.casa_warn = Table(np.array(self.casa_warn), names=['value', 'file', 'software_version'])
        self.casa_severe = Table(np.array(self.casa_severe), names=['value', 'file', 'software_version'])
        self.artip_errors = Table(np.array(self.artip_errors, dtype='object'), names=['value', 'file'])

    def visualize_errors(self):
        '''
        Make barplots of the errors
        '''
        # All errors
        plt.bar(['CASA SEVERE','CASA WARN','ARTIP error'],
                [len(self.casa_severe),len(self.casa_warn),len(self.artip_errors)])
        plt.title(self.card_name)
        plt.savefig(self.output+self.card_name+'_all.png')
        plt.close()
        logger.info('Saving '+self.output+self.card_name+'_all.png')

        # All severe errors
        severe_funcs = [line.split('\t')[2].split(':')[0] for line in self.casa_severe['value']]
        func, counts = np.unique(severe_funcs, return_counts=True)

        if len(counts) > 0:
            plt.bar(func,counts)
            plt.title(self.card_name+' SEVERE')
            plt.savefig(self.output+self.card_name+'_severe.png')
            plt.close()
            logger.info('Saving '+self.output+self.card_name+'_severe.png')

        # All warnings
        warn_funcs = [line.split('\t')[2].split(':')[0] for line in self.casa_warn['value']]
        func, counts = np.unique(warn_funcs, return_counts=True)

        if len(counts) > 0:
            plt.bar(func,counts)
            plt.title(self.card_name+' WARN')
            plt.savefig(self.output+self.card_name+'_warn.png')
            plt.close()
            logger.info('Saving '+self.output+self.card_name+'_warn.png')

    def match_errors(self):
        '''
        Match found errors to a dictionary of known errors and return the unknown errors
        '''
        logger.info('Comparing errors found to known errors')

        path = Path(__file__).parent / 'input/known_errors.json'
        with open(path) as f:
            template_dict = json.load(f)

        severe_template = template_dict[self.software_version]['SEVERE']
        warn_template = template_dict[self.software_version]['WARN']

        # Escape certain characters otherwise regex breaks
        severe_template = [re.escape(template) for template in severe_template]
        warn_template[1:] = [re.escape(template) for template in warn_template[1:]]

        warn_match = []
        for warn in self.casa_warn:
            warn_match.append(any([re.search(template,warn['value']) for template in warn_template]))

        severe_match = []
        for severe in self.casa_severe:
            severe_match.append(any([re.search(template,severe['value']) for template in severe_template]))

        unknown_warn = []
        unknown_severe = []
        if len(warn_match) > 0:
            unknown_warn = self.casa_warn[~np.array(warn_match)]
        if len(severe_match) > 0:
            unknown_severe = self.casa_severe[~np.array(severe_match)]

        return unknown_warn, unknown_severe

    def write_errors(self, warn, severe):
        '''
        Write errors to a file

        Keyword arguments:
        warn -- CASA warnings to be written to file
        severe -- CASA severe to be written to file
        '''
        logger.info('Writing errors to '+self.output+self.card_name+'_errors')
        with open(self.output+self.card_name+'_errors.txt','w') as f:
            if warn:
                f.write('**********CASA WARN************\n')
                warn_by_file = warn.group_by('file')
                for i, group in enumerate(warn_by_file.groups.indices[:-1]):
                    f.write(warn_by_file['file'][group]+'\n')
                    f.write('\n'.join(warn_by_file.groups[i]['value']))
                    f.write('\n')
            if severe:
                f.write('**********CASA SEVERE************\n')
                severe_by_file = severe.group_by('file')
                for i, group in enumerate(severe_by_file.groups.indices[:-1]):
                    f.write(severe_by_file[group]['file']+'\n')
                    f.write('\n'.join(severe_by_file.groups[i]['value']))
                    f.write('\n')
            if self.artip_errors:
                f.write('**********ARTIP ERROR************\n')
                # Always write all found artip errors to file
                for error in self.artip_errors:
                    f.write(error['file']+'\n')
                    f.write('\n'.join(error['value']))

    def check_fluxmodel(self):
        '''
        Parse the setjy and fluxscale tasks for the model fluxes of the calibrators
        '''
        setjy_log = self.casa_log[self.casa_log['task'] == 'FillCalibratorModel']['file']
        getjy_log = self.casa_log[self.casa_log['task'] == 'PhaseCalibration']['file']

        # Define the format strings
        getjy_format = ' Flux density for {field} in SpW=0 (freq={freq} Hz) is: {flux} +/- {fluxerr} (SNR = {SNR}, N = {N})'
        setjy_format1 = '  {field} (fld ind {fldind}) spw {spw}  [I={I}, Q={Q}, U={U}, V={V}] Jy @ {freq}Hz, ({ref})'
        setjy_format2 = '     {freq}         {I}'

        # Parse getjy files
        setjy = []
        setjy_lines = parse_file(setjy_log[0])

        for line in setjy_lines:
            if 'imager::setjy()\t  J' in line:
                clip_line = line.split('\t')[-1]
                setjy.append(match_format_string(setjy_format1,clip_line))
            if 'imager::setjy()\t     ' in line:
                clip_line = line.split('\t')[-1]
                setjy.append(match_format_string(setjy_format2,clip_line))

        # Parse fluxscale files
        fluxscale = []
        getjy_lines = parse_file(getjy_log[0])

        for line in getjy_lines:
            if 'fluxscale::::\t Flux density for' in line:
                clip_line = line.split('\t')[-1]
                fluxscale.append(match_format_string(getjy_format,clip_line))

        # Match gain calibrator to catalog
        path = Path(__file__).parent / 'input/Lband-gain-calibrators_HRKJW.csv'
        cal_cat = Table.read(path)
        for entry in fluxscale:
            source = cal_cat[cal_cat['source_name'] == entry['field']]
            if source:
                logger.info(f"Gain calibrator {entry['field']} with flux {entry['flux']}+/-{entry['fluxerr']} Jy,")
                logger.info(f"has been matched to the reference catalog, where the source has flux {source[0]['flux']} Jy.")

        # Combine the data in a dictionary and write to a file
        data = {
                "setjy": setjy,
                "getjy": fluxscale
        }

        logger.info('Saving model fluxes to '+self.output+'flux_model.json')
        with open(self.output+'flux_model.json', "w") as f:
            json.dump(data, f, indent=4)

    def check_flags(self):
        '''
        Check flagged data during self calibration stages
        '''
        selfcal_log = self.casa_log[self.casa_log['task'] == 'ContinuumImagingCont']['file']
        flag_format = '   G Jones: In: {in_flag} / {in_tot}   ({in_percent}%) --> Out: {out_flag} / {out_tot}   ({out_percent}%) ({caltable})'

        selfcal_lines = parse_file(selfcal_log[0])

        for line in selfcal_lines:
            if 'applycal::::\t   G Jones: In:' in line:
                clip_line = line.split('\t')[-1]
                flag_info = match_format_string(flag_format, clip_line)

                in_flag = float(flag_info['in_percent'])
                out_flag = float(flag_info['out_percent'])
                if out_flag - in_flag > 1.5:
                    logger.important('A large amount of data has been flagged during one of the')
                    logger.important(f'self calibration stages (from {in_flag:.2f}% to {out_flag:.2f}%)')
                    logger.important('Associated caltable: '+flag_info['caltable'])

        logger.info(f'Percent of flagged data at the end of self-calibration: {out_flag:.2f}%')

    def get_rms(self):
        '''
        Get theoretical sensitivity from the CASA log
        '''
        imaging_log = self.casa_log[self.casa_log['task'] == 'Imaging']['file']
        sensitivity_format = '[{image}][Taylor0] Theoretical sensitivity (Jy/bm):{sensitivity}'

        imaging_lines = parse_file(imaging_log[0])

        for line in imaging_lines:
            if 'task_tclean::SIImageStoreMultiTerm::calcSensitivity \t[' in line:
                clip_line = line.split('\t')[-1]
                sensitivity = match_format_string(sensitivity_format, clip_line)
                break

        theo_sens = sensitivity['sensitivity']
        logger.info(f'Theoretical sensitivity calculated as {theo_sens} Jy/beam')
        return theo_sens

def main():

    parser = new_argument_parser()
    args = parser.parse_args()

    dataset_path = args.dataset_path
    experiment = args.experiment
    selection = args.selection

    experiment_path = os.path.join(dataset_path,'experiment_'+str(experiment),'*artip_*')

    split = experiment_path.split('/')
    output_folder = split[-3]+'_'+split[-2]+'_logs_output/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Find cards and sort them in a sensible order
    artip_cards = sorted(glob.glob(experiment_path),
                         key=lambda experiment_path: '_'.join(experiment_path.rsplit('_',2)[1:]))

    if not artip_cards:
        logger.warn('No ARTIP blocks found, please check path carefully')
        sys.exit()

    logger.info('Found '+str(len(artip_cards))+' ARTIP blocks')

    for card in artip_cards:
        logs = Logs(card, output_folder)
        if logs.card_name is None:
            continue

        logger.info('------------------'+logs.card_name.upper()+'------------------')

        indexed = logs.index_logs(select=selection)

        if indexed:
            logs.read_errors()
            warn, severe = logs.match_errors()
            if 'artip_cal' in card:
                logs.check_fluxmodel()
            if 'artip_cont' in card and logs.casa_log[logs.casa_log['task'] == 'ContinuumImagingCont']:
                logs.check_flags()
                theoretical_sensitivity = logs.get_rms()

                rms_map.plot_rms_steps(os.path.join(logs.output_path,logs.dataset),
                                       theoretical_sensitivity,
                                       output_folder)

            logs.get_processing_time()
            if warn or severe or logs.artip_errors:
                logs.write_errors(warn, severe)
                #logs.visualize_errors()
                logger.important('Possibly important errors have been found and have been written')
                logger.important('to the output folder '+logs.output+', please check the files.')
                logger.important('If the errors are harmless, consider adding them to the known_errors')
                logger.important('file so that they will not be considered important in the future.')
            else:
                logger.info('No relevant errors found, no output file written')

    logger.handlers.clear()
    os.rename('log-eval.log', os.path.join(output_folder,'log-eval.log'))


def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("dataset_path",
                        help="Path to data.")
    parser.add_argument("experiment",
                        help="Experiment index.")
    parser.add_argument("-s", "--selection", default='last',
                        help="""Timerange of logs to select. Use 'all' to select all logs,
                                'last' to only select most recent logs for each run
                                (default = 'all').""")
    return parser

if __name__ == '__main__':
    main()