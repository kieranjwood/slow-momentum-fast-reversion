import multiprocessing
import os

from settings.default import COMMODITIES_TICKERS, CPD_OUTPUT_FOLDER_DEFAULT

N_WORKERS = len(COMMODITIES_TICKERS)

if not os.path.exists(CPD_OUTPUT_FOLDER_DEFAULT):
    os.mkdir(CPD_OUTPUT_FOLDER_DEFAULT)

all_processes = [
    f'python script_cpd_example.py "{ticker}" "{os.path.join(CPD_OUTPUT_FOLDER_DEFAULT, ticker + ".csv")}" "2000-01-01" "2019-12-31"'
    for ticker in COMMODITIES_TICKERS
]
process_pool = multiprocessing.Pool(processes=N_WORKERS)
process_pool.map(os.system, all_processes)
